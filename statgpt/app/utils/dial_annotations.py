"""Sending `custom_content.annotations` on a DIAL choice.

An annotation claims a `<cit data-id="...">` marker tag in the message text and says what the
tag cites: the file, its page, and the label of the pill the client draws in its place. A tag no
annotation claims is rendered as literal text, so the array has to reach the reader for the
citations to appear.

This module is a stand-in for an SDK method that does not exist yet, and is meant to be deleted
rather than maintained. `Choice` numbers every child it knows about from its own counter —
`create_stage` from `_last_stage_index`, `add_attachment` from `_last_attachment_index` — which
is why `OpenAiToDialStreamer._process_stage` and `_process_attachment` hand over a payload and
never an index. `aidial_sdk` (pinned `>=0.39.0,<0.40.0`) has no concept of an annotation, so
there is no counter to borrow and no method to hand the array to.

If the SDK grows a `Choice.add_annotation`, delete this module together with
`AnnotationIndexSpace` and everything that threads it (`ChainParametersConfig`, the
`ChainParameters` accessor, the streamer's constructor argument, and the two entry points that
create the counter), and call the method from `_process_annotations` the way
`_process_attachment` already calls `add_attachment`. That also returns the two things this
workaround gives up: the `opened`/`closed` guard that `send_chunk` skips, and a real `NullChoice`
no-op in place of the `isinstance` test in `send_annotations` below.
"""

import logging
from collections.abc import Sequence
from typing import Any

from aidial_sdk.chat_completion import Choice

# Unexported import path: `ArbitraryChunk` and its `BaseChunk` are outside `aidial_sdk`'s
# `__all__`, so this could change without deprecation. It is used because the SDK has no
# annotations API, and this is the same mechanism its own state and attachment chunks go out
# through. `choice.add_attachment` is not an alternative: it assigns attachment indexes from its
# own counter and would renumber the annotations.
from aidial_sdk.chat_completion.chunks import ArbitraryChunk

from statgpt.app.utils.dial_stages import ChoiceI

_log = logging.getLogger(__name__)


def send_annotations(choice: ChoiceI, annotations: Sequence[dict[str, Any]]) -> None:
    """Emit `annotations` as one streamed delta on `choice`.

    Annotations belong to a message, so they are always sent to a choice and never to a stage.
    The array may arrive before or after the text it annotates: the client resolves annotations
    only once the message has finished streaming.

    Does nothing but log on a choice that cannot stream (`NullChoice` in the MCP context), the
    same way `NullChoice.add_attachment` does.
    """
    if not isinstance(choice, Choice):
        _log.warning(
            "send_annotations() called on %s, which cannot stream — %d annotation(s) dropped.",
            type(choice).__name__,
            len(annotations),
        )
        return

    choice.send_chunk(
        ArbitraryChunk(
            {
                "choices": [
                    {
                        "index": choice.index,
                        "finish_reason": None,
                        "delta": {"custom_content": {"annotations": list(annotations)}},
                    }
                ],
                "usage": None,
            }
        )
    )
