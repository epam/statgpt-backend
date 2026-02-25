import logging
import os
import tempfile
import zipfile
from mimetypes import guess_type

import yaml
from fastapi import BackgroundTasks, HTTPException, status
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.sql.expression import func

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.audit.decorators import audit_action
from statgpt.admin.settings.exim import JobsConfig
from statgpt.common import utils
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import AuditActionType, AuditEntityType
from statgpt.common.services import ChannelSerializer, ChannelService
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import dial_core_factory
from statgpt.common.utils.elastic import ElasticSearchFactory
from statgpt.common.vectorstore import VectorStoreFactory

from .background_tasks import background_task

_log = logging.getLogger(__name__)


class AdminPortalChannelService(ChannelService):

    @staticmethod
    def _parse_integrity_error(
        data: schemas.ChannelBase | schemas.ChannelUpdate, e: IntegrityError
    ) -> None:
        _log.warning(e)

        if "UniqueViolationError" in str(e.orig):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Key deployment_id='{data.deployment_id}' already exists.",
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unknown db error"
        )

    @staticmethod
    async def _export_dial_file_to_folder(
        dial_file_path: str, folder_path: str, auth_context: AuthContext
    ) -> None:
        async with dial_core_factory(dial_settings.url, auth_context.api_key) as dial_core:
            content, _ = await dial_core.get_file_by_path(dial_file_path)

        target_file = os.path.join(folder_path, JobsConfig.DIAL_FILES_FOLDER, dial_file_path)
        utils.write_bytes(content, target_file)

    @staticmethod
    async def _import_dial_file_from_folder(
        file_path: str, dial_file_path: str, auth_context: AuthContext
    ) -> None:
        content = utils.read_bytes(file_path)
        mime_type = guess_type(file_path)[0]
        if not mime_type:
            mime_type = "application/octet-stream"
        async with dial_core_factory(dial_settings.url, auth_context.api_key) as dial_core:
            await dial_core.put_file(dial_file_path, mime_type, content)

    async def _create_channel_model(self, data: schemas.ChannelBase) -> models.Channel:
        item = models.Channel(
            title=data.title,
            description=data.description,
            deployment_id=data.deployment_id,
            details=data.details.model_dump(mode="json", by_alias=True),
            llm_model=data.llm_model,
        )

        try:
            self._session.add(item)
            await self._session.flush()
        except IntegrityError as e:
            self._parse_integrity_error(data, e)
        return item

    @audit_action(entity_type=AuditEntityType.CHANNEL, action_type=AuditActionType.CREATE)
    async def create_channel(self, data: schemas.ChannelBase) -> schemas.Channel:
        item = await self._create_channel_model(data)
        return ChannelSerializer.db_to_schema(item)

    @audit_action(entity_type=AuditEntityType.CHANNEL, action_type=AuditActionType.UPDATE)
    async def update(self, item_id: int, data: schemas.ChannelUpdate) -> schemas.Channel:
        item = await self._get_item_or_raise(item_id)

        if data.details is not None:
            data.details = data.details.model_dump(mode="json", by_alias=True)  # type: ignore

        query = (
            update(models.Channel)
            .where(models.Channel.id == item.id)
            .values(**data.model_dump(mode="json", exclude_unset=True), updated_at=func.now())
            .returning(models.Channel)
        )
        try:
            item = (await self._session.execute(query)).scalar_one()
            await self._session.flush()
        except IntegrityError as e:
            self._parse_integrity_error(data, e)

        return ChannelSerializer.db_to_schema(item)

    @audit_action(entity_type=AuditEntityType.CHANNEL, action_type=AuditActionType.DELETE)
    async def delete(self, item_id: int, auth_context: AuthContext) -> schemas.Channel:
        item = await self._get_item_or_raise(item_id)
        deleted_item = ChannelSerializer.db_to_schema(item)
        _log.info(f"Deleting {item}")

        await self._clear_vector_store(item, auth_context)

        if self.is_channel_hybrid(item):
            await self._clear_elastic_indexes(item)

        await self._session.delete(item)
        await self._session.flush()
        return deleted_item

    async def _clear_vector_store(self, channel: models.Channel, auth_context: AuthContext) -> None:
        vector_store_factory = VectorStoreFactory(session=self._session)

        collections = [
            channel.indicator_table_name,
            channel.available_dimensions_table_name,
            channel.special_dimensions_table_name,
        ]
        for collection in collections:
            vector_store = await vector_store_factory.get_vector_store(
                collection_name=collection,
                auth_context=auth_context,
                embedding_model_name=channel.llm_model,
            )
            await vector_store.clear()

    async def _clear_elastic_indexes(self, channel: models.Channel) -> None:
        _log.info(f"Clearing Elastic indexes for channel {channel!r}")
        for index in [channel.matching_index_name, channel.indicators_index_name]:
            await self._clear_elastic_index(index)

    @staticmethod
    async def _clear_elastic_index(index_name: str) -> None:
        """Drops index if it is possible, otherwise deletes all documents from the index."""

        try:
            index = await ElasticSearchFactory.get_index(index_name)
        except Exception:
            _log.exception(f"Failed to get Elastic index {index_name!r}")
            return

        try:
            await index.delete()
            _log.info(f"Dropped Elastic index {index_name!r}")
            return
        except Exception as e:
            _log.info(f"Exception: {e}")
            _log.warning(f"Failed to drop Elastic index {index_name!r}, trying to clear it...")

        try:
            res = await index.delete_by_query(query={"match_all": {}})
            _log.debug(res)
            _log.info(f"Cleared {res['deleted']} documents from Elastic index {index_name!r}")
        except Exception:
            _log.exception(f"Failed to clear Elastic index {index_name!r}")

    async def export_channel_to_folder(
        self,
        channel_id: int,
        folder_path: str,
        scope: schemas.ExportScope,
        auth_context: AuthContext,
    ) -> models.Channel:
        channel_db = await self.get_model_by_id(channel_id)
        channel_schema = ChannelSerializer.db_to_schema(channel_db)

        if scope.includes_configs():
            channel_file = os.path.join(folder_path, JobsConfig.CHANNEL_FILE)
            channel_data = channel_schema.model_dump(mode="json", include=JobsConfig.CHANNEL_FIELDS)
            utils.write_yaml(channel_data, channel_file)

        if (
            scope.includes_dial_files()
            and channel_schema.details.plain_content
            and channel_schema.details.plain_content.details.file_path
        ):
            await self._export_dial_file_to_folder(
                channel_schema.details.plain_content.details.file_path,
                folder_path,
                auth_context,
            )

        return channel_db

    async def import_channel_from_zip(
        self,
        zip_file: zipfile.ZipFile,
        clean_up: bool,
        scope: schemas.ExportScope,
        deployment_id: str | None,
        auth_context: AuthContext,
    ) -> models.Channel:
        if scope.includes_dial_files():
            await self._import_dial_files_from_zip(zip_file, auth_context)

        if scope.includes_configs():
            channel_data = await self._load_channel_data_from_zip(zip_file)
            if clean_up:
                await self._cleanup_existing_channel(channel_data.deployment_id, auth_context)
            created_channel = await self.create_channel(channel_data)
            await self._session.commit()
            return await self.get_model_by_id(created_channel.id)

        return await self._get_existing_channel_by_deployment_id(deployment_id)

    async def _import_dial_files_from_zip(
        self, zip_file: zipfile.ZipFile, auth_context: AuthContext
    ) -> None:
        _log.info("Uploading DIAL files for channel import")
        with tempfile.TemporaryDirectory() as temp_dir:
            members = [
                name
                for name in zip_file.namelist()
                if name.startswith(JobsConfig.DIAL_FILES_FOLDER)
            ]
            zip_file.extractall(temp_dir, members=members)

            dial_files_folder = os.path.join(temp_dir, JobsConfig.DIAL_FILES_FOLDER)
            for root, _, files in os.walk(dial_files_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    dial_file_path = os.path.relpath(file_path, dial_files_folder)
                    await self._import_dial_file_from_folder(
                        file_path, dial_file_path, auth_context
                    )

    async def _load_channel_data_from_zip(self, zip_file: zipfile.ZipFile) -> schemas.ChannelBase:
        with zip_file.open(JobsConfig.CHANNEL_FILE) as channel_file:
            channel_data_json = yaml.safe_load(channel_file.read())

        channel_data = schemas.ChannelBase.model_validate(channel_data_json)
        _log.info(f"Importing channel: {channel_data!r}")
        return channel_data

    async def _cleanup_existing_channel(
        self, deployment_id: str, auth_context: AuthContext
    ) -> None:
        try:
            existing_channel = await self.get_channel_by_deployment_id(deployment_id)
        except NoResultFound:
            pass  # No channel found, nothing to delete
        else:
            await self.delete(existing_channel.id, auth_context=auth_context)

    async def _get_existing_channel_by_deployment_id(
        self, deployment_id: str | None
    ) -> models.Channel:
        if deployment_id is None:
            raise ValueError("deployment_id is required when importing indexes only.")
        try:
            return await self.get_channel_by_deployment_id(deployment_id)
        except NoResultFound:
            raise ValueError(f"Channel with deployment_id {deployment_id} not found during import.")

    async def deduplicate_channel_dimensions(
        self, channel_id: int, auth_context: AuthContext
    ) -> None:
        """Deduplicates Available_Dimensions and Special_Dimensions vector stores for channel.

        Deduplication is performed based on document content.
        Documents with identical content will be merged.
        """
        channel = await self.get_model_by_id(channel_id)
        vector_store_factory = VectorStoreFactory(session=self._session)

        _log.info(f"Deduplicating available_dimensions for channel {channel_id}")
        available_dims_store = await vector_store_factory.get_vector_store(
            collection_name=channel.available_dimensions_table_name,
            auth_context=auth_context,
            embedding_model_name=channel.llm_model,
        )
        await available_dims_store.deduplicate_by_document_content()

        _log.info(f"Deduplicating special_dimensions for channel {channel_id}")
        special_dims_store = await vector_store_factory.get_vector_store(
            collection_name=channel.special_dimensions_table_name,
            auth_context=auth_context,
            embedding_model_name=channel.llm_model,
        )
        await special_dims_store.deduplicate_by_document_content()

        _log.info(f"Deduplication completed for channel {channel_id}")

    async def deduplicate_all_dimensions(
        self, background_tasks: BackgroundTasks, channel_id: int, auth_context: AuthContext
    ) -> schemas.Channel:
        """Deduplicates dimension vector stores for all datasets in a channel.

        Starts background deduplication task and returns the channel.
        """
        channel = await self.get_model_by_id(channel_id)

        background_tasks.add_task(
            deduplicate_dimensions_in_background_task,
            channel_id=channel_id,
            auth_context=auth_context,
        )

        return ChannelSerializer.db_to_schema(channel)


@background_task
async def deduplicate_dimensions_in_background_task(
    channel_id: int,
    auth_context: AuthContext,
) -> None:
    try:
        async with models.get_session_contex_manager() as session:
            await AdminPortalChannelService(session).deduplicate_channel_dimensions(
                channel_id=channel_id,
                auth_context=auth_context,
            )
    except Exception:
        _log.exception(f"Failed to deduplicate dimensions for channel {channel_id}")
