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
from common.utils.async_utils import catch_and_log_async
from statgpt.schemas.tool_artifact import DataQueryArtifact
from statgpt.utils import get_json_markdown, get_python_code_markdown


class DataQueryArtifactDisplayer:
    def __init__(
        self,
        choice: Choice,
        config: DataQueryAttachments,
        max_cells: int,
        auth_context: AuthContext,
    ):
        self._choice = choice
        self._config = config
        self._auth_context = auth_context
        self._max_cells = max_cells

    async def display(self, data_query_artifacts: list[DataQueryArtifact]) -> None:
        responses = self._merge_data_responses(data_query_artifacts)
        await self._display_data_responses(responses)

    async def get_system_message_content(
        self, data_query_artifacts: list[DataQueryArtifact]
    ) -> str:
        responses = self._merge_data_responses(data_query_artifacts)

        datasets_content = filter(
            None, [self._get_system_message_content(response) for response in responses.values()]
        )

        return "\n\n".join(datasets_content)

    def _get_system_message_content(self, response: DataResponse) -> str | None:
        df = response.visual_dataframe
        cells_number = df.shape[0] * df.shape[1] if df is not None else 0

        if cells_number == 0:
            return None
        elif cells_number <= self._max_cells:
            # convert df to markdown with full precision
            markdown_content = df.to_markdown()
            return (
                f"Data from dataset {response.dataset_name}: \n\n"
                + markdown_content
                + "\n\n The data itself is shown to user in the table view in the UI. When citing the data, make "
                + "sure to use full precision values from the table."
            )
        else:
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

    async def _attach_data_response(
        self, attachments_storage: AttachmentsStorage, data_response: DataResponse
    ) -> None:
        tasks = [
            self._attach_custom_table(attachments_storage, data_response),
            self._attach_plotly_grid(attachments_storage, data_response),
            self._attach_csv(attachments_storage, data_response),
            self._attach_markdown_json_query(data_response),
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

    @catch_and_log_async(logger)
    async def _attach_csv(
        self, attachments_storage: AttachmentsStorage, data_response: DataResponse
    ) -> dict[str, str] | None:
        if not self._config.csv_file.enabled or data_response.visual_dataframe is None:
            return None

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

        content = get_json_markdown(json.dumps(data, indent=2))
        title = data_response.enrich_attachment_name(self._config.json_query.name)
        return dict(type=MediaTypes.MARKDOWN, title=title, data=content)

    @catch_and_log_async(logger)
    async def _attach_json_query(self, data_response: DataResponse) -> dict[str, str] | None:
        if not self._config.json_query.enabled:
            return None

        data = data_response.json_query
        if data is None:
            return None

        content = json.dumps(data)
        title = data_response.enrich_attachment_name(self._config.json_query.name)
        return dict(type=MediaTypes.JSON, title=title, data=content)

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

        data = data_response.python_code
        if not data:
            return None

        title = data_response.enrich_attachment_name(self._config.python_code.name)
        return dict(type=MediaTypes.MARKDOWN, title=title, data=get_python_code_markdown(data))

    @catch_and_log_async(logger)
    async def _attach_plotly_grid(
        self, attachments_storage: AttachmentsStorage, data_response: DataResponse
    ) -> dict[str, str] | None:
        if not self._config.plotly_grid.enabled:
            return None

        data = data_response.plotly_grid
        if data is None:
            return None

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

            response = await attachments_storage.put_json(data_response.file_name, json.dumps(data))
            title = data_response.enrich_attachment_name(self._config.custom_table.name)
            return dict(type=MediaTypes.TTYD_TABLE, title=title, url=response.url)
        except Exception:
            logger.exception("Failed to attach custom table for dataset")
            return None
