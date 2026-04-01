import asyncio
import json
import string

import pandas as pd
import plotly.graph_objects as go
from aidial_sdk.chat_completion import Choice

from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.app.utils import get_json_markdown, get_python_code_markdown
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import logger
from statgpt.common.data.base import DataResponse
from statgpt.common.schemas.data_query_tool import DataQueryAttachments
from statgpt.common.schemas.enums import DataParsingStatus
from statgpt.common.utils import AttachmentsStorage, MediaTypes, attachments_storage_factory
from statgpt.common.utils.async_utils import catch_and_log_async

_DATA_DISCLAIMER = """
DISCLAIMER: Some of the values are coded, in that case for each column you will find two columns: {column}_ID and \
{column}_Name. Mind that numerical codes are **just identifiers** and do not have any numerical meaning.
"""
_PARSING_PARTIALLY_FAILED_DISCLAIMER = """
IMPORTANT: Some of the data could not be parsed correctly and is not included in the data shown below. \
It will be still visible to the user in the table view in the UI.
"""

_PYTHON_SDMX1_HEADER = """\
# Uses the [sdmx1 library](https://pypi.org/project/sdmx1/)
# Install with:
# ```bash
# pip install sdmx1
# ```

import sdmx"""


class DataQueryArtifactDisplayer:
    def __init__(
        self,
        choice: Choice,
        config: DataQueryAttachments,
        chat_config: StatGPTConfiguration,
        max_cells: int,
        auth_context: AuthContext,
    ):
        self._choice = choice
        self._config = config
        self._chat_config = chat_config
        self._auth_context = auth_context
        self._max_cells = max_cells

    async def display(self, data_query_artifacts: dict[str, DataQueryArtifact]) -> None:
        data_query_artifacts_list = list(data_query_artifacts.values())
        responses = self._merge_data_responses(data_query_artifacts_list)
        tasks = [self._display_data_responses(responses)]
        if self._chat_config.enable_debug_attachments:
            tasks.append(self._display_eval_attachments(data_query_artifacts))
        await asyncio.gather(*tasks)

    async def get_system_message_content(
        self, data_query_artifacts: list[DataQueryArtifact]
    ) -> str:
        responses = self._merge_data_responses(data_query_artifacts)

        datasets_content = [
            self._get_system_message_content(response) for response in responses.values()
        ]
        datasets_content_filtered = list(filter(None, datasets_content))

        return "\n\n".join(datasets_content_filtered)

    async def _display_eval_attachments(
        self, data_query_artifacts: dict[str, DataQueryArtifact]
    ) -> None:
        async with attachments_storage_factory(self._auth_context.api_key) as attachments_storage:
            tasks = []
            for tool_call_id, artifact in data_query_artifacts.items():
                tasks.append(
                    self._display_eval_attachment(tool_call_id, artifact, attachments_storage)
                )
            await asyncio.gather(*tasks)

    async def _display_eval_attachment(
        self,
        tool_call_id: str,
        artifact: DataQueryArtifact,
        attachments_storage: AttachmentsStorage,
    ) -> None:
        eval_attachment_content = artifact.eval_attachment.model_dump(mode="json")
        response = await self._attach_json_file(
            attachments_storage=attachments_storage,
            data=eval_attachment_content,
            filename=f"data_query_eval_attachment_{tool_call_id}.json",
            title=f"Data Query Eval data: {tool_call_id}",
            indent=2,
        )
        self._choice.add_attachment(**response)

    def _get_system_message_content(self, response: DataResponse) -> str | None:
        if response.status.parsing_status == DataParsingStatus.FAILED:
            return None

        df = response.visual_dataframe.copy()
        cells_number = df.shape[0] * df.shape[1] if df is not None else 0

        if cells_number == 0:
            return self._get_no_data_message(response)
        elif cells_number <= self._max_cells:
            try:
                return self._get_data_message(response, df)
            except Exception:
                logger.exception(
                    "Failed to convert dataframe to tsv, displaying error message instead, data response: %s",
                    response,
                )
                return self._get_data_display_error_message(response)
        else:
            return self._get_data_too_large_message(response, cells_number)

    @classmethod
    def _get_no_data_message(cls, response: DataResponse) -> str:
        return f"Data from dataset {response.dataset_name} is empty."

    @classmethod
    def _get_data_message(cls, response: DataResponse, df: pd.DataFrame) -> str:
        # add _ID suffix to all columns with _Name suffix variations
        to_rename = {}
        for col in df.columns:
            if isinstance(col, str) and col.endswith("_Name") and col[:-5] in df.columns:
                to_rename[col[:-5]] = col[:-5] + "_ID"
        df.rename(columns=to_rename, inplace=True)
        # convert df to tsv with full precision
        tsv = df.to_csv(
            sep="\t",
            index=False,
            na_rep="NA",
            date_format="%Y-%m-%d",
            lineterminator="\n",
        )

        result = (
            _DATA_DISCLAIMER
            + f"Data from dataset {response.dataset_name}: \n\n<DATA>\n"
            + tsv
            + "\n</DATA>\n\n The data itself is shown to user in the table view in the UI. If citing the data, "
            + "make sure to use full precision values from the table."
        )

        if response.status.parsing_status == DataParsingStatus.PARTIALLY_FAILED:
            result = _PARSING_PARTIALLY_FAILED_DISCLAIMER + "\n" + result
        return result

    @classmethod
    def _get_data_display_error_message(cls, response: DataResponse) -> str:
        return (
            f"Data from dataset {response.dataset_name} could not be displayed due to an error. "
            "The data itself is shown to user in the table view in the UI."
        )

    @classmethod
    def _get_data_too_large_message(cls, response: DataResponse, cells_number: int) -> str:
        return (
            f"Data from dataset {response.dataset_name} contains {cells_number} cells. "
            "The data is too large to include in the message. The data itself is shown to user in the table view "
            "in the UI."
        )

    def _merge_data_responses(
        self, data_query_artifacts: list[DataQueryArtifact]
    ) -> dict[str, DataResponse]:
        responses: dict[str, DataResponse] = {}

        for artifact in data_query_artifacts:
            for dataset_id, data_response in artifact.data_responses.items():
                if dataset_id not in responses:
                    responses[dataset_id] = data_response
                else:
                    responses[dataset_id] = responses[dataset_id].merge(data_response)
        return responses

    async def _display_data_responses(self, responses: dict[str, DataResponse]) -> None:
        tasks = []
        async with attachments_storage_factory(self._auth_context.api_key) as attachments_storage:
            for dataset_id, response in responses.items():
                tasks.append(self._attach_data_response(attachments_storage, response))
            await asyncio.gather(*tasks)

        if self._chat_config.merge_python_code:
            merged_python = await self._create_merged_python_code_attachment(responses)
            if merged_python is not None:
                self._choice.add_attachment(**merged_python)

    async def _attach_data_response(
        self, attachments_storage: AttachmentsStorage, data_response: DataResponse
    ) -> None:
        tasks = [
            self._attach_custom_table(attachments_storage, data_response),
            self._attach_plotly_grid(attachments_storage, data_response),
            self._attach_csv(attachments_storage, data_response),
            self._attach_markdown_json_query(data_response),
            self._attach_json_query(data_response),
        ]

        if not self._chat_config.merge_python_code:
            tasks.append(self._attach_python_code(data_response))

        if self._config.plotly_graphs.enabled:
            assert (
                self._config.plotly_graphs.name is not None
            ), "plotly_graphs.name must be set when enabled"
            for title, figure in data_response.get_plotly_graphs_with_names(
                self._config.plotly_graphs.name
            ):
                attachment_name = title.translate(str.maketrans("", "", string.punctuation))
                # There's a limit on the attachment name length. However, it is different on local setup
                # and in the dev environment.
                attachment_name = attachment_name.replace(" ", "_")[:32]
                tasks.append(
                    self._attach_plotly(
                        attachments_storage,
                        figure,
                        attachment_name,
                        title,
                    )
                )

        # Attachments can't be added in parallel, because the order of attachments is important.
        # But we can upload files to the Dial core in parallel.
        attachments = await asyncio.gather(*tasks)
        for attachment in attachments:
            if attachment is not None:
                self._choice.add_attachment(**attachment)

    @catch_and_log_async(logger)
    async def _attach_csv(
        self, attachments_storage: AttachmentsStorage, data_response: DataResponse
    ) -> dict[str, str] | None:
        if not self._config.csv_file.enabled or data_response.visual_dataframe is None:
            return None

        assert self._config.csv_file.name is not None, "csv_file.name must be set when enabled"
        response = await attachments_storage.put_csv_from_dataframe(
            data_response.file_name, data_response.visual_dataframe
        )
        title = data_response.enrich_attachment_name(self._config.csv_file.name)
        return dict(type=response.content_type, title=title, url=response.url)

    @catch_and_log_async(logger)
    async def _attach_markdown_json_query(
        self, data_response: DataResponse
    ) -> dict[str, str] | None:
        if not self._config.json_query.enabled:
            return None

        data = data_response.json_query_old
        if data is None:
            return None

        assert self._config.json_query.name is not None, "json_query.name must be set when enabled"
        content = get_json_markdown(json.dumps(data, ensure_ascii=False, indent=2))
        title = data_response.enrich_attachment_name(self._config.json_query.name)
        return dict(type=MediaTypes.MARKDOWN, title=title, data=content)

    @catch_and_log_async(logger)
    async def _attach_json_query(self, data_response: DataResponse) -> dict[str, str] | None:
        if not self._config.json_query.enabled:
            return None

        data = data_response.json_query
        if data is None:
            return None

        assert self._config.json_query.name is not None, "json_query.name must be set when enabled"
        content = json.dumps(data, ensure_ascii=False)
        title = data_response.enrich_attachment_name(self._config.json_query.name)
        return dict(type=MediaTypes.JSON, title=title, data=content)

    @catch_and_log_async(logger)
    async def _attach_json_file(
        self,
        attachments_storage: AttachmentsStorage,
        data: dict,
        filename: str,
        title: str,
        indent: int | None = None,
    ) -> dict[str, str]:
        json_content = json.dumps(data, ensure_ascii=False, indent=indent)
        response = await attachments_storage.put_json(filename, json_content)
        return dict(type=MediaTypes.JSON, title=title, url=response.url)

    @catch_and_log_async(logger)
    async def _attach_plotly(
        self,
        attachments_storage: AttachmentsStorage,
        figure: go.Figure,
        filename: str,
        title: str,
    ) -> dict[str, str]:
        chart_json = figure.to_json()
        response = await attachments_storage.put_json(filename, chart_json)
        return dict(type=MediaTypes.PLOTLY, title=title, url=response.url)

    @catch_and_log_async(logger)
    async def _attach_python_code(self, data_response: DataResponse) -> dict[str, str] | None:
        if not self._config.python_code.enabled:
            return None

        body = data_response.get_python_code_body()
        if not body:
            return None

        if self._config.python_code.name is None:
            raise ValueError("python_code.name must be set when enabled")
        code = _PYTHON_SDMX1_HEADER + "\n\n" + body
        title = data_response.enrich_attachment_name(self._config.python_code.name)
        return dict(type=MediaTypes.MARKDOWN, title=title, data=get_python_code_markdown(code))

    @catch_and_log_async(logger)
    async def _create_merged_python_code_attachment(
        self, responses: dict[str, DataResponse]
    ) -> dict[str, str] | None:
        if not self._config.merged_python_code.enabled:
            return None

        if self._config.merged_python_code.name is None:
            raise ValueError("merged_python_code.name must be set when enabled")

        items = [r for r in responses.values() if r.get_python_code_body()]
        if not items:
            return None

        if len(items) == 1:
            response = items[0]
            title = self._config.merged_python_code.name
            code = _PYTHON_SDMX1_HEADER + "\n\n" + response.get_python_code_body()  # type: ignore[operator]
            return dict(
                type=MediaTypes.MARKDOWN,
                title=title,
                data=get_python_code_markdown(code),
            )

        bodies: list[str] = []
        for i, response in enumerate(items, start=1):
            body = response.get_python_code_body(suffix=f"_{i}")
            if body:
                bodies.append(f"# Dataset: {response.dataset_name}\n{body}")

        if not bodies:
            return None

        merged_code = _PYTHON_SDMX1_HEADER + "\n\n" + "\n\n".join(bodies)
        title = self._config.merged_python_code.name
        return dict(
            type=MediaTypes.MARKDOWN,
            title=title,
            data=get_python_code_markdown(merged_code),
        )

    @catch_and_log_async(logger)
    async def _attach_plotly_grid(
        self, attachments_storage: AttachmentsStorage, data_response: DataResponse
    ) -> dict[str, str] | None:
        if not self._config.plotly_grid.enabled:
            return None

        data = data_response.plotly_grid
        if data is None:
            return None

        assert (
            self._config.plotly_grid.name is not None
        ), "plotly_grid.name must be set when enabled"
        title = data_response.enrich_attachment_name(self._config.plotly_grid.name)
        return await self._attach_plotly(attachments_storage, data, data_response.file_name, title)

    @catch_and_log_async(logger)
    async def _attach_custom_table(
        self, attachments_storage: AttachmentsStorage, data_response: DataResponse
    ) -> dict[str, str] | None:
        if not self._config.custom_table.enabled:
            return None

        try:
            data = data_response.custom_table_dict
            if not data:
                return None

            assert (
                self._config.custom_table.name is not None
            ), "custom_table.name must be set when enabled"
            response = await attachments_storage.put_json(
                data_response.file_name, json.dumps(data, ensure_ascii=False)
            )
            title = data_response.enrich_attachment_name(self._config.custom_table.name)
            return dict(type=MediaTypes.TTYD_TABLE, title=title, url=response.url)
        except Exception:
            logger.exception("Failed to attach custom table for dataset")
            return None
