import os
import tempfile
import zipfile
from mimetypes import guess_type

import yaml
from fastapi import HTTPException, status
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.sql.expression import func

import common.models as models
import common.schemas as schemas
from admin_portal.settings.exim import JobsConfig
from common import utils
from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.services import ChannelSerializer, ChannelService
from common.settings.dial import dial_settings
from common.utils import dial_core_factory
from common.vectorstore import VectorStoreFactory


class AdminPortalChannelService(ChannelService):

    @staticmethod
    def _parse_integrity_error(
        data: schemas.ChannelBase | schemas.ChannelUpdate, e: IntegrityError
    ) -> None:
        logger.warning(e)

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
            content, content_type = await dial_core.get_file_by_path(dial_file_path)

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
            await self._session.commit()
        except IntegrityError as e:
            self._parse_integrity_error(data, e)
        return item

    async def create_channel(self, data: schemas.ChannelBase) -> schemas.Channel:
        item = await self._create_channel_model(data)
        return ChannelSerializer.db_to_schema(item)

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
            await self._session.commit()
        except IntegrityError as e:
            self._parse_integrity_error(data, e)

        return ChannelSerializer.db_to_schema(item)

    async def delete(self, item_id: int, auth_context: AuthContext) -> None:
        item = await self._get_item_or_raise(item_id)
        logger.info(f"Deleting {item}")

        await self._clear_vector_store(item, auth_context)

        await self._session.delete(item)
        await self._session.commit()

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

    async def export_channel_to_folder(
        self, channel_id: int, folder_path: str, auth_context: AuthContext
    ) -> models.Channel:
        channel_db = await self.get_model_by_id(channel_id)
        channel_schema = ChannelSerializer.db_to_schema(channel_db)

        if (
            channel_schema.details.plain_content
            and channel_schema.details.plain_content.details.file_path
        ):
            await self._export_dial_file_to_folder(
                channel_schema.details.plain_content.details.file_path, folder_path, auth_context
            )

        channel_file = os.path.join(folder_path, JobsConfig.CHANNEL_FILE)
        channel_data = channel_schema.model_dump(mode="json", include=JobsConfig.CHANNEL_FIELDS)
        utils.write_yaml(channel_data, channel_file)
        return channel_db

    async def import_channel_from_zip(
        self, zip_file: zipfile.ZipFile, clean_up: bool, auth_context: AuthContext
    ) -> models.Channel:
        # Read all files from JobsConfig.DIAL_FILES_FOLDER recursively and upload them to DIAL

        logger.info("Uploading DIAL files for channel import")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract all files that start with the dial files folder path
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

        with zip_file.open(JobsConfig.CHANNEL_FILE) as channel_file:
            channel_data_json = yaml.safe_load(channel_file.read())

        channel_data = schemas.ChannelBase.model_validate(channel_data_json)
        logger.info(f"Importing channel: {channel_data!r}")

        if clean_up:
            try:
                existing_channel = await self.get_channel_by_deployment_id(
                    channel_data.deployment_id
                )
            except NoResultFound:
                pass  # No channel found, nothing to delete
            else:
                await self.delete(existing_channel.id, auth_context=auth_context)

        return await self._create_channel_model(channel_data)
