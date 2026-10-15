"""Relaying the stages of a sub-deployment to the user's message (#749).

Since aidial-sdk 0.45 a deployment can nest a stage under another one: the first chunk of the
stage names its parent in `parent_stage_index`. The streamer rebuilds that tree on the response,
under the indexes the response gave the stages, and falls back to the root of the choice when the
parent cannot take a child.
"""

import asyncio
import contextlib
import itertools
import logging
from typing import Any

import pytest
from aidial_sdk.chat_completion import Choice
from aidial_sdk.chat_completion.chunks import BaseChunk
from openai.types.chat import ChatCompletionChunk

from statgpt.app.utils.dial_stages import NullChoice
from statgpt.app.utils.openai_to_dial_streamer import OpenAiToDialStreamer
from statgpt.common.schemas import StagesConfig


def _open_choice() -> Choice:
    """A real SDK choice that already holds a stage of its own, like a tool-call stage.

    The stage shifts the indexes the response gives the relayed stages away from the indexes the
    sub-deployment gave them, so a parent index relayed verbatim lands on the wrong stage.
    """
    choice = Choice(asyncio.Queue(), 0)
    choice.open()
    choice.create_stage("Deep Research").open()
    return choice


def _streamer(choice: Any) -> OpenAiToDialStreamer:
    return OpenAiToDialStreamer(
        choice,
        choice,
        deployment="deep-research",
        show_debug_stages=False,
        stages_config=StagesConfig(debug_only=False),
        annotation_index_space=itertools.count(),
    )


def _opened(index: int, name: str, parent: int | None = None) -> dict[str, Any]:
    """The delta of the sub-deployment that opens a stage."""
    stage: dict[str, Any] = {"index": index, "name": name, "status": None}
    if parent is not None:
        stage["parent_stage_index"] = parent
    return stage


def _completed(index: int, status: str = "completed") -> dict[str, Any]:
    """The delta of the sub-deployment that closes a stage."""
    return {"index": index, "status": status}


def _chunk(*stages: dict[str, Any]) -> ChatCompletionChunk:
    """A completion chunk of the sub-deployment carrying stage deltas."""
    return ChatCompletionChunk.model_validate(
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "deep-research",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {"custom_content": {"stages": list(stages)}},
                }
            ],
        }
    )


def _sent_stages(choice: Choice) -> list[dict[str, Any]]:
    """Every stage delta the choice's queue holds, in the order sent."""
    sent = []
    while not choice._queue.empty():
        queued = choice._queue.get_nowait()
        # The queue also carries the end-of-stream and error markers, which have no payload.
        if not isinstance(queued, BaseChunk):
            continue
        for ch in queued.to_dict().get("choices", []):
            sent.extend(ch.get("delta", {}).get("custom_content", {}).get("stages", []))
    return sent


def _tree(sent: list[dict[str, Any]]) -> dict[str, str | None]:
    """The name of the parent of every stage opened on the choice, keyed by the stage's name."""
    names = {stage["index"]: stage["name"] for stage in sent if "name" in stage}
    return {
        stage["name"]: names.get(stage.get("parent_stage_index"))
        for stage in sent
        if "name" in stage
    }


def _closed(sent: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The name and status of every stage closed on the choice, in the order they were closed."""
    names = {stage["index"]: stage["name"] for stage in sent if "name" in stage}
    return [
        (names[stage["index"]], stage["status"])
        for stage in sent
        if stage.get("status") is not None
    ]


class TestStageTree:
    def test_relays_a_two_level_tree(self):
        choice = _open_choice()

        with _streamer(choice) as streamer:
            streamer.send_chunk(_chunk(_opened(0, "Plan")))
            streamer.send_chunk(_chunk(_opened(1, "Search", parent=0)))
            streamer.send_chunk(_chunk(_opened(2, "Read report", parent=0)))
            streamer.send_chunk(_chunk(_completed(1), _completed(2), _completed(0)))
            streamer.send_chunk(_chunk(_opened(3, "Answer")))

        assert _tree(_sent_stages(choice)) == {
            "Deep Research": None,
            "Plan": None,
            "Search": "Plan",
            "Read report": "Plan",
            "Answer": None,
        }

    def test_child_of_a_parent_never_relayed_opens_at_the_root(self, caplog):
        choice = _open_choice()

        with caplog.at_level(logging.WARNING), _streamer(choice) as streamer:
            streamer.send_chunk(_chunk(_opened(0, "Plan")))
            streamer.send_chunk(_chunk(_opened(1, "Search", parent=7)))

        assert _tree(_sent_stages(choice))["Search"] is None
        assert "parent stage 7 is unknown or already closed" in caplog.text

    def test_child_of_a_closed_parent_opens_at_the_root(self):
        choice = _open_choice()

        with _streamer(choice) as streamer:
            streamer.send_chunk(_chunk(_opened(0, "Plan")))
            streamer.send_chunk(_chunk(_completed(0)))
            streamer.send_chunk(_chunk(_opened(1, "Search", parent=0)))

        assert _tree(_sent_stages(choice))["Search"] is None

    @pytest.mark.parametrize("status", ["completed", "failed"])
    def test_relays_the_status_a_stage_closes_with(self, status):
        choice = _open_choice()

        with _streamer(choice) as streamer:
            streamer.send_chunk(_chunk(_opened(0, "Plan")))
            streamer.send_chunk(_chunk(_completed(0, status)))

        assert _closed(_sent_stages(choice)) == [("Plan", status)]

    @pytest.mark.parametrize(
        ("error", "status"), [(None, "completed"), (RuntimeError("the stream broke"), "failed")]
    )
    def test_exit_closes_children_before_their_parents(self, error, status):
        """The sub-deployment stopped streaming with its stages open: the streamer closes them.

        They close as `failed` when the stream broke, and as `completed` otherwise.
        """
        choice = _open_choice()

        with contextlib.suppress(RuntimeError), _streamer(choice) as streamer:
            streamer.send_chunk(_chunk(_opened(0, "Plan")))
            streamer.send_chunk(_chunk(_opened(1, "Search", parent=0)))
            streamer.send_chunk(_chunk(_opened(2, "Read report", parent=1)))
            if error is not None:
                raise error

        assert _closed(_sent_stages(choice)) == [
            ("Read report", status),
            ("Search", status),
            ("Plan", status),
        ]

    def test_nested_stages_relay_to_a_null_choice(self):
        """The MCP context has no response to relay stages to: they must be dropped silently."""
        with _streamer(NullChoice()) as streamer:
            streamer.send_chunk(_chunk(_opened(0, "Plan")))
            streamer.send_chunk(_chunk(_opened(1, "Search", parent=0)))
            streamer.send_chunk(_chunk({"index": 1, "content": "Searching..."}, _completed(1)))
