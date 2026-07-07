"""Eval-pipeline artifact writer for mcp_lite.

Adds one tool — `write_data_query_artifact` — that the agent MUST call at the
end of every data-query session. It emits the same JSON format the eval
pipeline consumes (mirror of MR !1142 in statgpt-eval), extended to support
multi-dataset queries (the `dimension_id_to_name` dict can carry multiple
dataset_uuid keys).

Experiment metadata comes from env vars set at server startup — the agent
doesn't need to know them. Timings in the emitted artifact are placeholder
(`start_time_utc=end_time_utc=NOW`, `response_time=0`); the orchestrator that
launched the subagent post-edits the JSON with real wall-clock + token usage
from the harness's task-notification before running the eval pipeline.
"""

import json
import logging
import os
import uuid as uuid_mod
from datetime import datetime, timezone
from typing import Annotated

from fastmcp.exceptions import ToolError

from ._provider import mcp_tools

_log = logging.getLogger(__name__)

_TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
_JOBS_METADATA_FILENAME = "job_stats.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime(_TS_FORMAT)


def _env_or_error(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise ToolError(
            f"Environment variable {name!r} is not set. The MCP server must be "
            "started in eval mode with EVAL_EXPERIMENT_ID, EVAL_EXPERIMENT_NAME, "
            "and EVAL_JOBS_DIR all defined."
        )
    return val


@mcp_tools.tool
def write_data_query_artifact(
    test_case_id: Annotated[
        str,
        "Opaque test-case id supplied to you by the runtime — copy it verbatim from the "
        "session metadata you were given. Used to match this artifact to its ground-truth "
        "target in the eval pipeline.",
    ],
    query: Annotated[str, "The verbatim user question this artifact answers."],
    dataset_uuid_to_urn: Annotated[
        dict[str, str],
        "Mapping of internal dataset UUID -> dataset URN (e.g. 'BIS:WS_NA_SEC_DSS(1.0)'). "
        "Include every dataset you selected as a candidate answer; multi-dataset queries "
        "have multiple entries.",
    ],
    dimension_id_to_name: Annotated[
        dict[str, dict[str, dict[str, str]]],
        "Selected dimension mapping per dataset: "
        "{dataset_uuid: {dimension_id: {term_id: term_name}}}. "
        "Keys must match `dataset_uuid_to_urn`. For each dataset, include the "
        "dimension codes you would pass to `execute_sdmx_query.selection` to "
        "answer the question. Skip TIME_PERIOD — it's filtered separately.",
    ],
    n_availability_queries: Annotated[
        int,
        "Total count of `availability_query` calls you made while building this answer.",
    ] = 0,
) -> str:
    """**CALL AT THE END OF EVERY DATA-QUERY SESSION.** Records your selected
    dataset(s) and dimension code mapping into a job artifact that the eval
    pipeline scores against the ground truth.

    Required steps before calling:
    1. Identify candidate dataset(s) via `search_indicators` / `list_datasets`.
    2. Map dimension codes via `search_codes` / `dataset_structure` and the
       `dimensions` payload returned by `search_indicators`.
    3. Verify each selection is reachable via `availability_query`.
    4. (Optional) `execute_sdmx_query` to sanity-check the data exists.
    5. Call THIS tool with the dim mapping keyed by dataset UUID.

    Returns the absolute path of the written artifact JSON file. The session is
    considered complete after this call.
    """
    output_dir = _env_or_error("EVAL_JOBS_DIR")
    experiment_id = _env_or_error("EVAL_EXPERIMENT_ID")
    experiment_name = _env_or_error("EVAL_EXPERIMENT_NAME")

    if set(dimension_id_to_name) != set(dataset_uuid_to_urn):
        raise ToolError(
            "Keys of `dimension_id_to_name` must match `dataset_uuid_to_urn` exactly. "
            f"Missing: {set(dataset_uuid_to_urn) - set(dimension_id_to_name)}; "
            f"Extra: {set(dimension_id_to_name) - set(dataset_uuid_to_urn)}"
        )

    job_id = uuid_mod.uuid4().hex[:22]
    ts = _utc_now()
    tool_call_id = "claude_data_query_0"

    artifact = {
        "test_case": {
            "id": test_case_id,
            "name": test_case_id,
            "tags": [],
            "comments": "",
            "conversation": [{"role": "user", "content": query, "target": None}],
        },
        "responses": [
            {
                "data": {
                    "id": f"claude-{test_case_id}",
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "custom_content": {
                                    "state": {
                                        "tool_messages": [
                                            {
                                                "type": "ai",
                                                "content": "",
                                                "tool_calls": [
                                                    {
                                                        "name": "Query_Data",
                                                        "args": {"query": query},
                                                        "id": tool_call_id,
                                                        "type": "tool_call",
                                                    }
                                                ],
                                            },
                                            {
                                                "type": "tool",
                                                "tool_call_id": tool_call_id,
                                                "status": "success",
                                                "content": "success",
                                                "custom_content": {
                                                    "state": {
                                                        "type": "DATA_QUERY",
                                                        "dimension_id_to_name": dimension_id_to_name,
                                                        "indexed_datasets_id_map": dataset_uuid_to_urn,
                                                        "datasets_selection_response": {
                                                            "dataset_ids": list(dataset_uuid_to_urn)
                                                        },
                                                        "n_availability_queries": n_availability_queries,
                                                    }
                                                },
                                            },
                                        ]
                                    }
                                },
                            }
                        }
                    ],
                },
                # Orchestrator post-edits these with real values from the harness
                # task-notification (duration_ms, total_tokens).
                "status_code": 200,
                "start_time_utc": ts,
                "end_time_utc": ts,
                "response_time": 0.0,
                "exception_info": None,
                "failed_attempts": 0,
            }
        ],
    }

    os.makedirs(output_dir, exist_ok=True)

    artifact_fp = os.path.join(output_dir, f"{job_id}.json")
    with open(artifact_fp, "w") as f:
        json.dump(artifact, f, indent=2)

    job_meta = {
        "job_id": job_id,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "time_start": ts,
        "time_end": ts,
        "status": "success",
        "exception_info": None,
        "test_case": {
            "id": test_case_id,
            "name": test_case_id,
            "tags": [],
            "comments": "",
            "conversation": [{"role": "user", "content": query, "target": None}],
        },
    }

    stats_fp = os.path.join(output_dir, _JOBS_METADATA_FILENAME)
    with open(stats_fp, "a") as f:
        f.write(json.dumps(job_meta) + "\n")

    _log.info(f"wrote artifact {artifact_fp} for test_case_id={test_case_id}")
    return artifact_fp
