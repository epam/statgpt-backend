import asyncio
import json

from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.app.utils.dial_stages import ChoiceI
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import logger
from statgpt.common.utils import AttachmentsStorage, MediaTypes, attachments_storage_factory
from statgpt.common.utils.async_utils import catch_and_log_async


class DataQueryEvalAttachmentsDisplayer:
    """Attaches the Data Query eval (debug) files to the choice.

    Kept apart from `DataQueryArtifactDisplayer` because it needs nothing from the channel's
    attachments config: the direct tool calls path returns these files without any of the data
    attachments, and need not have the data query tool configured at all.
    """

    def __init__(self, choice: ChoiceI, auth_context: AuthContext, enabled: bool):
        self._choice = choice
        self._auth_context = auth_context
        self._enabled = enabled

    @catch_and_log_async(logger)
    async def display(self, artifacts: dict[str, DataQueryArtifact]) -> None:
        """Attach one eval file per tool call, plus a discovery file where the lookup ran.

        Debug-only output must never cost the user their answer, so the whole method is
        wrapped: a failure to reach the attachments storage is logged and dropped.
        """
        if not self._enabled or not artifacts:
            return

        async with attachments_storage_factory(self._auth_context.api_key) as attachments_storage:
            tasks = [
                self._display_one(tool_call_id, artifact, attachments_storage)
                for tool_call_id, artifact in artifacts.items()
            ]
            await asyncio.gather(*tasks)

    async def _display_one(
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
        if response is not None:
            self._choice.add_attachment(**response)

        # The discovery lookup runs beside the query, so it reports in a file of its own.
        discovery = artifact.discovery_datasets_eval_attachment
        if discovery is None:
            return
        discovery_response = await self._attach_json_file(
            attachments_storage=attachments_storage,
            data=discovery.model_dump(mode="json"),
            filename=f"discovery_datasets_eval_attachment_{tool_call_id}.json",
            title=f"Discovery Datasets Eval data: {tool_call_id}",
            indent=2,
        )
        if discovery_response is not None:
            self._choice.add_attachment(**discovery_response)

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
