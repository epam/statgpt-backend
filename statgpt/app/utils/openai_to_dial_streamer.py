import json
import re
from typing import Any

from aidial_sdk.chat_completion import Stage
from openai.types.chat import ChatCompletionChunk

from statgpt.app.utils.dial_stages import ChoiceI
from statgpt.common.schemas import StagesConfig
from statgpt.common.schemas.token_usage import TokenUsageItem
from statgpt.common.utils.token_usage_context import get_token_usage_manager

# Deep Research echoes the forwarded query as a leading, fully quoted `Query: "..."` line. Match
# that exact shape (not any `query:`-prefixed line) so a legitimate response that merely opens with
# "Query: ..." is never clipped from the display.
_LEADING_QUERY_ECHO_RE = re.compile(r'^query:\s*".*"$', re.IGNORECASE)


class OpenAiToDialStreamer:
    def __init__(
        self,
        target: ChoiceI | Stage,
        choice: ChoiceI,
        deployment: str,
        show_debug_stages: bool,
        stages_config: StagesConfig,
        stream_content: bool = True,
        strip_leading_query: bool = False,
    ) -> None:
        """Creates a streamer that processes OpenAI ChatCompletionChunks and sends them to Dial.

        Args:
            target: Choice or Stage object to append content and attachments to.
            choice: Choice object to create new stages.
            deployment: Deployment id or name that will be used to track token usage.
            stream_content: If True, the content will be appended to the `target` as it is received.
            stream_stages: If True, the stages will be created with the content and attachments from the chunks.
            strip_leading_query: If True, a leading ``Query: "..."`` echo line is dropped from the
                displayed content. Only the live display is filtered; ``content`` still returns the
                deployment's output verbatim.
        """

        self._target = target
        self._choice = choice
        self._deployment = deployment
        self._show_debug_stages = show_debug_stages
        self._stages_config = stages_config
        self._stream_content = stream_content
        self._strip_leading_query = strip_leading_query

        self._content = ""
        self._stages: dict[int, Stage] = {}
        self._attachments: list[dict[str, Any]] = []
        self._state: dict[str, Any] | None = None
        # Display-only buffer for the leading `Query: "..."` echo strip. Holds the very first
        # streamed chunks until the first line is complete, so it can be dropped if it is an echo.
        self._leading_resolved = not strip_leading_query
        self._leading_buffer = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._flush_leading_buffer()
        self._exit_opened_stages(exc_type, exc_val, exc_tb)
        return False

    @property
    def content(self) -> str:
        return self._content

    @property
    def attachments(self) -> list[dict[str, Any]]:
        return self._attachments

    @property
    def state(self) -> dict[str, Any] | None:
        """The `custom_content.state` set by the downstream deployment, if any."""
        return self._state

    @property
    def attachments_metadata(self) -> str:
        """Returns basic metadata of attachments in JSON format."""
        res = []
        for attachment in self._attachments:
            a_title = attachment.get('title')
            a_type = attachment.get('type') or attachment.get('reference_type')
            res.append({"title": a_title, "type": a_type})
        return json.dumps(res)

    @property
    def content_with_attachments_metadata(self) -> str:
        if not self._attachments:
            return self.content

        return (
            f"{self.content}\n\n### Metadata of attached files:\n\n"
            f"```json\n{self.attachments_metadata}\n```"
        )

    def send_chunk(self, chunk: ChatCompletionChunk) -> None:
        for ch in chunk.choices:
            if content := ch.delta.content:
                self._process_content(content)

            if custom_content := getattr(ch.delta, 'custom_content', None):
                self._process_custom_content(custom_content)

        if usages := getattr(chunk, 'statistics', {}).get('usage_per_model'):
            self._update_token_usage(usages)

    def _process_content(self, content: str) -> None:
        # `_content` is always the deployment's verbatim output (used for state/session replay);
        # only the live display is filtered.
        self._content += content

        if self._stream_content:
            self._append_display(content)

    def _append_display(self, content: str) -> None:
        """Append streamed content to the display target, dropping a leading ``Query: "..."``
        echo when `strip_leading_query` is set."""
        if not self._leading_resolved:
            self._leading_buffer += content
            # Wait until the first non-empty line is fully received (a newline follows it),
            # so a chunk boundary can't split the `Query:` line mid-decision.
            if "\n" not in self._leading_buffer.lstrip():
                return
            content = self._take_leading_buffer()

        if content:
            self._target.append_content(content)

    def _take_leading_buffer(self) -> str:
        """Resolve the buffered first line: drop it (and the blank line after it) if it is a
        ``Query: "..."`` echo, then return the remaining buffered content to display."""
        self._leading_resolved = True
        buffer, self._leading_buffer = self._leading_buffer, ""
        first_line, _, remainder = buffer.lstrip().partition("\n")
        if _LEADING_QUERY_ECHO_RE.match(first_line.strip()):
            return remainder.lstrip("\n")
        return buffer

    def _flush_leading_buffer(self) -> None:
        """Emit any content still buffered for the leading-query strip when the stream ends
        without a trailing newline (e.g. a one-line message)."""
        if self._stream_content and not self._leading_resolved and self._leading_buffer:
            if content := self._take_leading_buffer():
                self._target.append_content(content)

    def _process_custom_content(self, custom_content: dict[str, Any]) -> None:
        if (state := custom_content.get('state')) is not None:
            self._state = state

        if attachments := custom_content.get('attachments'):
            for attachment in attachments:
                self._process_attachment(attachment)

        if not self._stages_config.debug_only or self._show_debug_stages:
            for stage in custom_content.get('stages', []):
                self._process_stage(stage)

    def _process_attachment(self, attachment: dict[str, Any]) -> None:
        if attachment.get('data') is None and attachment.get('url') is None:
            attachment['data'] = ''

        self._attachments.append(attachment)
        if self._stream_content:
            self._target.add_attachment(
                type=attachment.get('type'),
                title=attachment.get('title'),
                data=attachment.get('data'),
                url=attachment.get('url'),
                reference_url=attachment.get('reference_url'),
                reference_type=attachment.get('reference_type'),
            )

    def _process_stage(self, stage: dict[str, Any]) -> None:
        index = stage['index']
        name = stage.get('name')

        if index not in self._stages:
            self._stages[index] = self._choice.create_stage(name or '')
            self._stages[index].open()
        elif name:
            self._stages[index].append_name(name)

        if content := stage.get('content'):
            self._stages[index].append_content(content)

        if attachments := stage.get('attachments'):
            for attachment in attachments:
                self._stages[index].add_attachment(
                    type=attachment.get('type'),
                    title=attachment.get('title'),
                    data=attachment.get('data'),
                    url=attachment.get('url'),
                    reference_url=attachment.get('reference_url'),
                    reference_type=attachment.get('reference_type'),
                )

        if stage.get('status') == 'completed':
            self._stages[index].close()

    def _exit_opened_stages(self, exc_type, exc_val, exc_tb) -> None:
        for stage in self._stages.values():
            stage.__exit__(exc_type, exc_val, exc_tb)

    def _update_token_usage(self, usages: list[dict[str, int | str]]) -> None:
        token_usage_manager = get_token_usage_manager()

        for usage in usages:
            token_usage_manager.add_usage(
                TokenUsageItem(
                    deployment=self._deployment,
                    model=usage['model'],
                    prompt_tokens=usage['prompt_tokens'],
                    completion_tokens=usage['completion_tokens'],
                )
            )
