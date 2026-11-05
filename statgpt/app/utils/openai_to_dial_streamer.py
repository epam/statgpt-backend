import json
import logging
from collections.abc import Iterator
from typing import Any

from aidial_sdk.chat_completion import Stage, Status
from openai.types.chat import ChatCompletionChunk

from statgpt.app.utils.citation_ids import (
    Citation,
    CitationIdSpace,
    find_tag_ids,
    get_source_quote,
    get_source_title,
    get_tag_id,
    replace_tag_ids,
    with_tag_id,
)
from statgpt.app.utils.custom_content_rewriter import CustomContentRewriter
from statgpt.app.utils.dial_annotations import send_annotations
from statgpt.app.utils.dial_stages import ChoiceI
from statgpt.common.schemas import StagesConfig
from statgpt.common.schemas.token_usage import TokenUsageItem
from statgpt.common.utils.token_usage_context import get_token_usage_manager

_log = logging.getLogger(__name__)

# Hands out the index this response gives a relayed annotation. One counter per response,
# created with `itertools.count()` and shared by every streamer of that response: see
# `OpenAiToDialStreamer._renumber_annotation`. A counter rather than a map keyed by streamer,
# so that the shared object outlives no streamer and none of their buffered reports.
#
# It exists only because the SDK owns no annotation index. `Choice` numbers stages and
# attachments from its own counters, which is why `_process_stage` and `_process_attachment`
# below pass a payload and never an index; annotations have no such method, so the numbering
# falls to us. A `Choice.add_annotation` upstream would delete this alias and everything that
# threads it — see the module docstring of `dial_annotations` for the full list.
AnnotationIndexSpace = Iterator[int]

# Heads the citations note of a tool response (see `OpenAiToDialStreamer.citations_note`). Says
# how to use the tags in the note itself, since a channel may replace the part of the system
# prompt that says it too.
_CITATIONS_NOTE_HEADER = (
    '### Sources of the citation tags:\n\n'
    'When your answer uses a statement cited above, copy its citation tag verbatim right after'
    ' it: never change, shorten or invent a tag. Do not copy this list into your answer.'
)


class OpenAiToDialStreamer:
    def __init__(
        self,
        target: ChoiceI | Stage,
        choice: ChoiceI,
        deployment: str,
        show_debug_stages: bool,
        stages_config: StagesConfig,
        annotation_index_space: AnnotationIndexSpace,
        citation_id_space: CitationIdSpace,
        stream_content: bool = True,
        rewriter: CustomContentRewriter | None = None,
    ) -> None:
        """Creates a streamer that processes OpenAI ChatCompletionChunks and sends them to Dial.

        Args:
            target: Choice or Stage object to append content and attachments to.
            choice: Choice object to create new stages, and to send annotations to.
            deployment: Deployment id or name that will be used to track token usage.
            annotation_index_space: The annotation index counter of the whole response, shared
                with every other streamer of the same response. Required rather than defaulted,
                so a new call site fails loudly instead of numbering annotations on its own.
            citation_id_space: The citation ids of the whole response, shared with every other
                streamer of the same response. Required for the same reason.
            stream_content: If True, the content will be appended to the `target` as it is received.
                The streamed content keeps the citation ids of the sub-deployment, which match no
                relayed annotation, so stream only to a stage: DIAL Chat resolves no citation tag
                in a stage anyway.
            rewriter: If set, rewrites the annotations and attachments (including stage
                attachments) before they are relayed.
            stream_stages: If True, the stages will be created with the content and attachments from the chunks.
        """

        self._target = target
        self._choice = choice
        self._deployment = deployment
        self._show_debug_stages = show_debug_stages
        self._stages_config = stages_config
        self._annotation_index_space = annotation_index_space
        self._citation_id_space = citation_id_space
        self._stream_content = stream_content
        self._rewriter = rewriter

        self._content = ""
        # The relayed stages, keyed by the index the deployment gave them.
        self._stages: dict[int, Stage] = {}
        # The keys of those still open: only an open stage can take a child.
        self._open_stages: set[int] = set()
        self._attachments: list[dict[str, Any]] = []
        self._annotations: list[dict[str, Any]] = []
        self._state: dict[str, Any] | None = None
        # This streamer's own annotations only: the index its sub-deployment gave one, mapped
        # to the index this response gave it. Dies with the streamer.
        self._annotation_indexes: dict[int, int] = {}
        # Likewise for citations: the tag id its sub-deployment gave one, mapped to the id this
        # response gave it.
        self._citation_ids: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_opened_stages(exc_type, exc_val, exc_tb)
        if exc_type is None:
            self._warn_of_unmatched_citations()
        return False

    @property
    def content(self) -> str:
        """The content, its citation tags renumbered like the relayed annotations claiming them."""
        return replace_tag_ids(self._content, self._citation_ids)

    @property
    def raw_content(self) -> str:
        """The content as the deployment sent it, its citation tags not renumbered. Replay this,
        not `content`, to the deployment: the ids of `content` are not ids it gave its tags."""
        return self._content

    @property
    def attachments(self) -> list[dict[str, Any]]:
        return self._attachments

    @property
    def annotations(self) -> list[dict[str, Any]]:
        """The annotations relayed to the choice, as sent (rewritten and renumbered)."""
        return self._annotations

    @property
    def citations(self) -> list[Citation]:
        """Every citation tag the relayed annotations claim, with the source it cites and the
        passage it quotes.

        Each distinct combination of tag id, title and quote is listed once, in the order first
        seen, so each entry matches one card in the pill's popup. Annotations are not merged by
        `index`, as Deep Research does not re-send one to extend it. An annotation that names no
        source (see `get_source_title`) is listed with no title, so the agent is not left with an
        id it cannot look up.
        """
        # One key per distinct (tag id, title, quote) combination. Each tuple is built from a single
        # annotation, so its three values always belong together. `dict.fromkeys` drops a repeated
        # key, and a dict keeps insertion order, so each entry stays where it was first seen.
        # The keys are tuples because a plain pydantic model is not hashable.
        keys = dict.fromkeys(
            (tag_id, get_source_title(annotation), get_source_quote(annotation))
            for annotation in self._annotations
            if (tag_id := get_tag_id(annotation)) is not None
        )
        return [Citation(id=tag_id, title=title, quote=quote) for tag_id, title, quote in keys]

    def citations_note(self, *, quotes: bool) -> str:
        """The source of every citation tag of the content, to append to a tool response so the
        agent can tell which source backs a statement. Empty if the content cites nothing.

        The note starts with how to use the tags, so the agent learns it whatever the system
        prompt of the channel says. A citation that names no source leaves the title out; the
        rest of the annotation, such as the cited file and page box, is left out to keep the
        response small. The annotations relayed to the client are not affected.

        Args:
            quotes: Whether to give the passage each citation quotes. A quote is often a long page
                excerpt, and the tool response stays in the agent's history for the rest of the
                conversation, so give them only where the title alone does not tell the cited
                passage, such as the dataset query a Deep Research citation quotes.
        """
        citations = self.citations
        if not quotes:
            # Without their quotes, two citations of one tag may read the same: list them once.
            citations = [
                Citation(id=tag_id, title=title)
                for tag_id, title in dict.fromkeys((c.id, c.title) for c in citations)
            ]
        if not citations:
            return ''
        citations_json = json.dumps(
            [c.model_dump(exclude_none=True) for c in citations], ensure_ascii=False
        )
        return f'\n\n{_CITATIONS_NOTE_HEADER}\n\n```json\n{citations_json}\n```'

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
        # Kept verbatim (see `raw_content`): `content` renumbers its citation tags on every read.
        self._content += content

        if self._stream_content:
            # TODO: renumber the citation tags streamed here once DIAL Chat resolves the tags of a
            # stage (epam/ai-dial-chat#9257). They keep the ids of the sub-deployment, which match
            # no relayed annotation, and the annotations claiming them may not have arrived yet.
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

        attachment = self._rewrite_attachment(attachment)
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
        is relayed unchanged except `index`, the id of the claimed tag, and the fields modified
        by the `rewriter`.
        """
        relayed = [
            self._renumber_citation(self._renumber_annotation(self._rewrite_annotation(annotation)))
            for annotation in annotations
        ]
        self._annotations.extend(relayed)
        send_annotations(self._choice, relayed)

    def _rewrite_annotation(self, annotation: dict[str, Any]) -> dict[str, Any]:
        if self._rewriter is None:
            return annotation
        return self._rewriter.rewrite_annotation(annotation)

    def _rewrite_attachment(self, attachment: dict[str, Any]) -> dict[str, Any]:
        if self._rewriter is None:
            return attachment
        return self._rewriter.rewrite_attachment(attachment)

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

    def _renumber_citation(self, annotation: dict[str, Any]) -> dict[str, Any]:
        """Move the id of the tag the annotation claims into the id space of the response.

        The agent copies the tags of a tool response into its answer, and the client resolves
        each tag to the annotations claiming its id. The ids of a sub-deployment are long hashes
        or counters that start over on every call, so they are swapped for `citation001-k3f9`, ...
        here and in `content` alike (see `citation_ids`). Every annotation claiming one tag keeps
        claiming one tag. An annotation that claims no tag is relayed unchanged.

        Like `_renumber_annotation`, this contains no `await` and needs no lock.
        """
        tag_id = get_tag_id(annotation)
        if tag_id is None:
            return annotation

        if tag_id not in self._citation_ids:
            self._citation_ids[tag_id] = self._citation_id_space.next_id()
        return with_tag_id(annotation, self._citation_ids[tag_id])

    def _warn_of_unmatched_citations(self) -> None:
        """Log the claimed tags that `content` did not renumber.

        Their annotations went out with the new id while the text kept the old one, so the client
        shows each such tag as literal text instead of a pill. Either the sub-deployment wrote a
        tag that `replace_tag_ids` does not recognise, or it claimed a tag it never wrote.
        """
        if unmatched := self._citation_ids.keys() - find_tag_ids(self._content):
            _log.warning(
                "%d citation tag(s) claimed by the annotations of %s are not in its content,"
                " so their pills cannot be resolved: %s",
                len(unmatched),
                self._deployment,
                sorted(unmatched),
            )

    def _process_stage(self, stage: dict[str, Any]) -> None:
        index = stage['index']
        name = stage.get('name')

        if index not in self._stages:
            self._open_stage(index, name or '', stage.get('parent_stage_index'))
        elif name:
            self._stages[index].append_name(name)

        if content := stage.get('content'):
            # TODO: likewise for the citation tags of the sub-deployment's own stages, relayed
            # with their ids unchanged (see `_process_content`, epam/ai-dial-chat#9257).
            self._stages[index].append_content(content)

        if attachments := stage.get('attachments'):
            for attachment in map(self._rewrite_attachment, attachments):
                self._stages[index].add_attachment(
                    type=attachment.get('type'),
                    title=attachment.get('title'),
                    data=attachment.get('data'),
                    url=attachment.get('url'),
                    reference_url=attachment.get('reference_url'),
                    reference_type=attachment.get('reference_type'),
                )

        # Relay the status the deployment closed the stage with: a stage left open here would be
        # closed as `completed` on exit, so a step that failed downstream would show as done.
        if (status := stage.get('status')) is not None:
            self._stages[index].close(Status(status))
            self._open_stages.discard(index)

    def _open_stage(self, index: int, name: str, parent_index: int | None) -> None:
        """Open the relayed stage under the relayed stage the deployment nested it in.

        A stage the deployment opened at the root opens at the root of the choice. So does a
        stage whose parent this streamer never relayed or has already closed: the SDK refuses to
        open a stage under such a parent, and a stage out of place is better than a failed
        response.
        """
        if parent_index is None:
            stage = self._choice.create_stage(name)
        elif parent_index in self._open_stages:
            stage = self._stages[parent_index].create_stage(name)
        else:
            _log.warning(
                "Relaying the stage %s of `%s` at the root: its parent stage %s is unknown"
                " or already closed",
                index,
                self._deployment,
                parent_index,
            )
            stage = self._choice.create_stage(name)

        stage.open()
        self._stages[index] = stage
        self._open_stages.add(index)

    def _exit_opened_stages(self, exc_type, exc_val, exc_tb) -> None:
        # A child is always relayed after its parent, so the reverse order closes every child
        # before its parent: the SDK warns about a stage closed while a child of it is open.
        for stage in reversed(self._stages.values()):
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
