"""`attachments_target` decides whether the RAG attachments are attached to the tool-result
stage (the default) or to the assistant message. The answer itself always stays on the stage.
"""

from typing import Any

import pytest

from statgpt.app.chains.file_rags.dial_rag import DialRagAgentFactory
from statgpt.app.chains.file_rags.file_rag_tool import _RAG_IMPLEMENTATIONS
from statgpt.common.schemas import AttachmentsTarget
from statgpt.common.schemas import FileRagTool as FileRagToolConfig
from statgpt.common.schemas import RAGVersion
from statgpt.common.schemas.tool_details import FileRagDetails


class FakeSink:
    """Records `add_attachment` calls; stands in for both a Stage and a Choice."""

    def __init__(self) -> None:
        self.attachments: list[dict[str, Any]] = []

    def add_attachment(self, **kwargs: Any) -> None:
        self.attachments.append(kwargs)


def _factory(
    attachments_target: AttachmentsTarget, attachment_url_override: str | None = None
) -> DialRagAgentFactory:
    tool_config = FileRagToolConfig(
        name="File_RAG",
        description="publications",
        details=FileRagDetails(
            version=RAGVersion.GENERIC,
            attachments_target=attachments_target,
            attachment_url_override=attachment_url_override,
        ),
    )
    return _RAG_IMPLEMENTATIONS[RAGVersion.GENERIC](tool_config, channel_config=None)  # type: ignore[arg-type]


def test_stage_is_the_default_sink():
    assert FileRagDetails(version=RAGVersion.GENERIC).attachments_target is AttachmentsTarget.stage


@pytest.mark.parametrize(
    "attachments_target, expects_choice",
    [(AttachmentsTarget.stage, False), (AttachmentsTarget.choice, True)],
)
def test_sink_follows_attachments_target(attachments_target, expects_choice):
    factory = _factory(attachments_target)
    target, choice = FakeSink(), FakeSink()

    sink = factory._attachments_sink(target, choice)  # type: ignore[arg-type]

    assert sink is (choice if expects_choice else target)


def test_url_override_is_applied_whichever_sink_is_used():
    attachments = [{"title": "doc", "reference_url": "files/bucket/folder/doc.pdf"}]

    for attachments_target in AttachmentsTarget:
        factory = _factory(attachments_target, attachment_url_override="http://public/files")
        sink = FakeSink()

        factory._append_attachments(sink, attachments)  # type: ignore[arg-type]

        (sent,) = sink.attachments
        assert sent["title"] == "doc"
        assert sent["reference_url"] == "http://public/files/doc.pdf"
