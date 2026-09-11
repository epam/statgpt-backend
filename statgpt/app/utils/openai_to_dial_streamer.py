import json
from collections.abc import Iterator
from typing import Any

from aidial_sdk.chat_completion import Stage
from openai.types.chat import ChatCompletionChunk

from statgpt.app.utils.dial_annotations import send_annotations
from statgpt.app.utils.dial_stages import ChoiceI
from statgpt.common.schemas import StagesConfig
from statgpt.common.schemas.token_usage import TokenUsageItem
from statgpt.common.utils.token_usage_context import get_token_usage_manager

# Hands out the index this response gives a relayed annotation. One counter per response,
# created with `itertools.count()` and shared by every streamer of that response: see
# `OpenAiToDialStreamer._renumber_annotation`. A counter rather than a map keyed by streamer,
# so that the shared object outlives no streamer and none of their buffered reports.
AnnotationIndexSpace = Iterator[int]


class OpenAiToDialStreamer:
    def __init__(
        self,
        target: ChoiceI | Stage,
        choice: ChoiceI,
        deployment: str,
        show_debug_stages: bool,
        stages_config: StagesConfig,
        annotation_index_space: AnnotationIndexSpace,
        stream_content: bool = True,
    ) -> None:
        """Creates a streamer that processes OpenAI ChatCompletionChunks and sends them to Dial.

        Args:
            target: Choice or Stage object to append content and attachments to.
            choice: Choice object to create new stages, and to send annotations to.
            deployment: Deployment id or name that will be used to track token usage.
            annotation_index_space: The annotation index counter of the whole response, shared
                with every other streamer of the same response. Required rather than defaulted,
                so a new call site fails loudly instead of numbering annotations on its own.
            stream_content: If True, the content will be appended to the `target` as it is received.
            stream_stages: If True, the stages will be created with the content and attachments from the chunks.
        """

        self._target = target
        self._choice = choice
        self._deployment = deployment
        self._show_debug_stages = show_debug_stages
        self._stages_config = stages_config
        self._annotation_index_space = annotation_index_space
        self._stream_content = stream_content

        self._content = ""
        self._stages: dict[int, Stage] = {}
        self._attachments: list[dict[str, Any]] = []
        self._state: dict[str, Any] | None = None
        # This streamer's own annotations only: the index its sub-deployment gave one, mapped
        # to the index this response gave it. Dies with the streamer.
        self._annotation_indexes: dict[int, int] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
        # `_content` is always the deployment's verbatim output (used for state/session replay).
        self._content += content

        if self._stream_content:
            self._target.append_content(content)

    def _process_custom_content(self, custom_content: dict[str, Any]) -> None:
        if (state := custom_content.get('state')) is not None:
            self._state = state

        if attachments := custom_content.get('attachments'):
            for attachment in attachments:
                self._process_attachment(attachment)

        if annotations := custom_content.get('annotations'):
            self._process_annotations(annotations)

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

    def _process_annotations(self, annotations: list[dict[str, Any]]) -> None:
        """Relay the sub-deployment's annotations to the user's message.

        They go to `_choice` and never to `_target`, which is a stage on some paths: an
        annotation claims a marker tag in a message, and a stage is not a message. Every field
        is relayed unchanged except `index`.
        """
        send_annotations(
            self._choice, [self._renumber_annotation(annotation) for annotation in annotations]
        )

    def _renumber_annotation(self, annotation: dict[str, Any]) -> dict[str, Any]:
        """Move the annotation's `index` into the index space of the whole response.

        Each sub-deployment numbers its own annotations from zero, and several of them run
        concurrently in one response (the Supreme Agent and the direct-tool-calls chain both
        dispatch tool calls with `asyncio.gather`), so the incoming numbers collide. The client
        compares indexes before it consults the tag id, and two annotations that share an index
        are read as one entry — which would hide the sources of a pill that cites several.

        An annotation that arrives without an index keeps none: the array it belongs to is
        either fully indexed or not indexed at all.

        `_annotation_indexes` maps this streamer's incoming indexes and no one else's, so an
        index that two sub-deployments both used stays two separate entries. Keeping the
        numbers those entries resolve to apart is the shared counter's job.

        This lookup deliberately contains no `await`, so concurrent tool calls cannot interleave
        inside it and it needs no lock. Keep it synchronous for that reason.
        """
        index = annotation.get('index')
        if index is None:
            return annotation

        if index not in self._annotation_indexes:
            self._annotation_indexes[index] = next(self._annotation_index_space)
        return {**annotation, 'index': self._annotation_indexes[index]}

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
