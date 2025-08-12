import asyncio
import json
import string

import plotly.graph_objects as go
from aidial_sdk.chat_completion import Choice

from common.auth.auth_context import AuthContext
from common.config import logger
from common.data.base import DataResponse
from common.schemas.data_query_tool import DataQueryAttachments
from common.utils import AttachmentsStorage, MediaTypes, attachments_storage_factory
from statgpt.schemas.tool_artifact import DataQueryArtifact
from statgpt.utils import get_json_markdown, get_python_code_markdown


class DataQueryArtifactDisplayer:
    def __init__(self, choice: Choice, config: DataQueryAttachments, auth_context: AuthContext):
        self._choice = choice
        self._config = config
        self._auth_context = auth_context

    async def display(self, data_query_artifacts: list[DataQueryArtifact]) -> None:
        responses = self._merge_data_responses(data_query_artifacts)
        await self._display_data_responses(responses)

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

    async def _attach_data_response(
        self, attachments_storage: AttachmentsStorage, data_response: DataResponse
    ) -> None:
        tasks = [
            self._attach_custom_table(attachments_storage, data_response),
            self._attach_plotly_grid(attachments_storage, data_response),
            self._attach_csv(attachments_storage, data_response),
            self._attach_json_query(data_response),
            self._attach_python_code(data_response),
        ]

        if self._config.plotly_graphs.enabled:
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

    async def _attach_csv(
        self, attachments_storage: AttachmentsStorage, data_response: DataResponse
    ) -> dict[str, str] | None:
        if not self._config.csv_file and data_response.visual_dataframe is None:
            return None

        response = await attachments_storage.put_csv_from_dataframe(
            data_response.file_name, data_response.visual_dataframe
        )
        title = data_response.enrich_attachment_name(self._config.csv_file.name)
        return dict(type=response.content_type, title=title, url=response.url)

    async def _attach_json_query(self, data_response: DataResponse) -> dict[str, str] | None:
        if not self._config.json_query:
            return None

        data = data_response.json_query
        if data is None:
            return None

        content = get_json_markdown(json.dumps(data, indent=2))
        title = data_response.enrich_attachment_name(self._config.json_query.name)
        return dict(type=MediaTypes.MARKDOWN, title=title, data=content)

    @classmethod
    async def _attach_plotly(
        cls,
        attachments_storage: AttachmentsStorage,
        figure: go.Figure,
        filename: str,
        title: str,
    ) -> dict[str, str]:
        chart_json = figure.to_json()
        response = await attachments_storage.put_json(filename, chart_json)
        return dict(type=MediaTypes.PLOTLY, title=title, url=response.url)

    async def _attach_python_code(self, data_response: DataResponse) -> dict[str, str] | None:
        if not self._config.python_code:
            return None

        data = data_response.python_code
        if not data:
            return None

        title = data_response.enrich_attachment_name(self._config.python_code.name)
        return dict(type=MediaTypes.MARKDOWN, title=title, data=get_python_code_markdown(data))

    async def _attach_plotly_grid(
        self, attachments_storage: AttachmentsStorage, data_response: DataResponse
    ) -> dict[str, str] | None:
        if not self._config.plotly_grid:
            return None

        data = data_response.plotly_grid
        if data is None:
            return None

        title = data_response.enrich_attachment_name(self._config.plotly_grid.name)
        return await self._attach_plotly(attachments_storage, data, data_response.file_name, title)

    async def _attach_custom_table(
        self, attachments_storage: AttachmentsStorage, data_response: DataResponse
    ) -> dict[str, str] | None:
        if not self._config.custom_table:
            return None

        try:
            data = data_response.custom_table_dict
            if not data:
                return None

            response = await attachments_storage.put_json(data_response.file_name, json.dumps(data))
            title = data_response.enrich_attachment_name(self._config.custom_table.name)
            return dict(type=MediaTypes.TTYD_TABLE, title=title, url=response.url)
        except Exception:
            logger.exception("Failed to attach custom table for dataset")
            return None
