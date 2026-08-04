import asyncio
import logging
import os
import zipfile
from typing import Any, Iterable, Sequence

import yaml
from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import func

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.audit.decorators import audit_action
from statgpt.admin.settings.exim import JobsConfig
from statgpt.common import utils
from statgpt.common.data import DataManager, DataSourceConfig
from statgpt.common.data.statgpt_sdmx_proxy.config_client import (
    ProxyConfigServerError,
    ProxyConfigValidationError,
)
from statgpt.common.schemas import AuditActionType, AuditEntityType
from statgpt.common.services import DataSourceSerializer, DataSourceService, DataSourceTypeService

_log = logging.getLogger(__name__)


class AdminPortalDataSourceService(DataSourceService):

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, None)  # No need for session lock in Admin Portal

    async def _parse_details_field(self, type_id: int, details: dict[str, Any]) -> DataSourceConfig:

        config_class = await DataSourceTypeService(self._session).get_config_class(type_id)

        try:
            parsed_config = config_class(**details)
        except ValidationError as e:
            _log.info(e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Failed to parse 'details' field: {e}",
            )
        except Exception as e:
            _log.info(e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Failed to parse 'details' field",
            )
        return parsed_config

    @staticmethod
    def _external_details_owner(
        item: schemas.DataSource,
    ) -> tuple[DataSourceConfig, str] | None:
        """Return the item's config and the key of the system owning part of its `details`.

        None when the data source type has no externally-owned details, or when `details`
        cannot be parsed - a data source must stay readable either way.
        """

        config_class = DataManager.get_config_class(item.type.name)
        try:
            config = config_class.model_validate(item.details)
        except ValidationError as e:
            _log.warning(
                "Cannot resolve the external details owner of data source %r: %s", item.title, e
            )
            return None

        if (key := config.external_details_key()) is None:
            return None
        return config, key

    @staticmethod
    async def _load_external_details(
        config: DataSourceConfig, key: str
    ) -> tuple[bool, dict[str, Any]]:
        """Load externally-owned details, reporting failures instead of raising.

        Returns:
            A `(loaded, details)` pair. `loaded` is False when the owning system could not be
            read, which is distinct from it answering that it holds no value yet.
        """

        try:
            return True, await config.load_external_details()
        except ProxyConfigServerError as e:
            _log.warning("Could not load the externally-owned details from %s: %s", key, e)
            return False, {}

    async def _enrich_with_external_details(self, items: Sequence[schemas.DataSource]) -> None:
        """Populate the externally-owned part of each data source's `details` from its owner.

        The owning system holds the value, so it is always read live instead of being cached or
        persisted. Data sources sharing an owner are read once. When an owner cannot be reached
        the keys are left absent - a data source must stay readable when its owner is down, and
        absent keys tell the admin portal not to offer those fields for editing.
        """

        targets = [(item, owner) for item in items if (owner := self._external_details_owner(item))]
        if not targets:
            return

        # One representative config per owner: configs sharing a key read the same value.
        by_key = {key: config for _, (config, key) in targets}
        keys = list(by_key)
        results = await asyncio.gather(
            *(self._load_external_details(by_key[key], key) for key in keys)
        )
        loaded = dict(zip(keys, results))

        for item, (_, key) in targets:
            was_loaded, details = loaded[key]
            if was_loaded:
                item.details.update(details)

    @staticmethod
    async def _push_external_details(config: DataSourceConfig) -> dict[str, Any] | None:
        """Send the externally-owned part of the submitted `details` back to its owner.

        Returns the value the owner confirmed it stored.

        None when the submitted `details` carry no externally-owned value: that means "leave the
        owning system alone", so a data source can be updated while its owner is unreachable
        without wiping the stored value. Once a value *is* submitted the owner is the only place
        it can live, so a failed push fails the whole write rather than dropping it silently.
        """

        try:
            return await config.push_external_details()
        except ProxyConfigValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
            ) from e
        except ProxyConfigServerError as e:
            # The owning system is a dependency we write through, so its failure is a gateway
            # failure - not this API being unavailable.
            _log.warning("Could not update the externally-owned details: %s", e)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    async def _apply_external_details(
        self, item: schemas.DataSource, pushed: dict[str, Any] | None
    ) -> None:
        """Complete a written data source's `details` with its externally-owned part.

        What the owner confirmed on the write is used as-is: it is what this action set, so
        neither the response nor its audit record can drift from it through a racing read-back.
        The owner is only read when the write left it untouched.
        """

        if pushed is None:
            await self._enrich_with_external_details([item])
        else:
            item.details.update(pushed)

    async def get_schema_by_id_with_external_details(self, item_id: int) -> schemas.DataSource:
        item = await self.get_schema_by_id(item_id)
        await self._enrich_with_external_details([item])
        return item

    async def get_data_sources_schemas_with_external_details(
        self, limit: int | None, offset: int
    ) -> list[schemas.DataSource]:
        items = await self.get_data_sources_schemas(limit=limit, offset=offset)
        await self._enrich_with_external_details(items)
        return items

    @audit_action(entity_type=AuditEntityType.DATA_SOURCE, action_type=AuditActionType.CREATE)
    async def create_data_source(self, data: schemas.DataSourceBase) -> schemas.DataSource:
        parsed_config = await self._parse_details_field(data.type_id, data.details)
        pushed_details = await self._push_external_details(parsed_config)

        item = models.DataSource(
            title=data.title,
            description=data.description,
            type_id=data.type_id,
            details=parsed_config.dump_for_storage(),
        )

        self._session.add(item)
        await self._session.flush()

        await self._session.refresh(item, attribute_names=["type"])
        result = DataSourceSerializer.db_to_schema(item)
        await self._apply_external_details(result, pushed_details)
        return result

    async def export_data_sources(
        self, data_sources: Iterable[schemas.DataSource], res_dir: str
    ) -> None:
        # Enrichment writes into `details`, and the caller keeps using these models after the
        # export, so work on copies.
        sources = [source.model_copy(deep=True) for source in data_sources]
        await self._enrich_with_external_details(sources)

        data = [
            source.model_dump(mode='json', include=JobsConfig.DATA_SOURCE_FIELDS)
            for source in sources
        ]
        data_sources_file = os.path.join(res_dir, JobsConfig.DATA_SOURCES_FILE)
        utils.write_yaml({'dataSources': data}, data_sources_file)

    @staticmethod
    def _details_changed(stored: schemas.DataSource, incoming_details: dict[str, Any]) -> bool:
        """Whether the imported `details` differ from what the data source already has.

        Both sides go through the type's config class so that defaults and field aliases are
        normalized, and so that a configuration owned by an external system is only compared
        when the imported payload carries one.
        """

        config_class = DataManager.get_config_class(stored.type.name)
        try:
            incoming_config = config_class.model_validate(incoming_details)
            stored_config = config_class.model_validate(stored.details)
        except ValidationError as e:
            _log.warning(
                "Cannot compare 'details' of data source %r, assuming they changed: %s",
                stored.title,
                e,
            )
            return True
        return not incoming_config.matches_stored(stored_config)

    async def import_data_sources_from_zip(
        self, zip_file: zipfile.ZipFile, update_data_sources: bool
    ) -> dict[str, schemas.DataSource]:
        existing_data_sources = {
            ds.title: ds
            for ds in await self.get_data_sources_schemas_with_external_details(
                limit=None, offset=0
            )
        }

        with zip_file.open(JobsConfig.DATA_SOURCES_FILE) as file:
            data_sources_json = yaml.safe_load(file)

        data_sources = {}
        for ds in data_sources_json['dataSources']:
            data_source_data = schemas.DataSourceBase.model_validate(ds)
            _log.info(f"Importing data source: {data_source_data!r}")

            if data_source := existing_data_sources.get(data_source_data.title):
                if update_data_sources:
                    data = {
                        field: getattr(data_source_data, field)
                        for field in schemas.DataSourceUpdate.model_fields.keys()
                        if field != 'details'
                        and getattr(data_source_data, field) != getattr(data_source, field)
                    }
                    if self._details_changed(data_source, data_source_data.details):
                        data['details'] = data_source_data.details
                    if data:
                        _log.info(f"Updating data source '{data_source_data.title}' with {data}")
                        data_source = await self.update(
                            data_source.id, schemas.DataSourceUpdate(**data)
                        )
                    else:
                        _log.info(f"Data source '{data_source_data.title}' exists and up-to-date.")
                else:
                    _log.info(f"Data source '{data_source_data.title}' already exists. Skipping.")
            else:
                data_source = await self.create_data_source(data_source_data)
            data_sources[data_source.title] = data_source

        return data_sources

    @audit_action(entity_type=AuditEntityType.DATA_SOURCE, action_type=AuditActionType.UPDATE)
    async def update(self, item_id: int, data: schemas.DataSourceUpdate) -> schemas.DataSource:

        item = await self._get_item_or_raise(item_id)

        pushed_details = None
        if data.details:
            parsed_config = await self._parse_details_field(item.type_id, data.details)
            pushed_details = await self._push_external_details(parsed_config)
            data.details = parsed_config.dump_for_storage()

        query = (
            update(models.DataSource)
            .where(models.DataSource.id == item.id)
            .values(**data.model_dump(exclude_unset=True), updated_at=func.now())
            .returning(models.DataSource)
        )
        item = (await self._session.execute(query)).scalar_one()
        await self._session.flush()

        await self._session.refresh(item, attribute_names=["type"])
        result = DataSourceSerializer.db_to_schema(item)
        await self._apply_external_details(result, pushed_details)
        return result

    @audit_action(entity_type=AuditEntityType.DATA_SOURCE, action_type=AuditActionType.DELETE)
    async def delete(self, item_id: int) -> schemas.DataSource:
        item = await self._get_item_or_raise(item_id)
        await self._session.refresh(item, attribute_names=["type"])
        deleted_item = DataSourceSerializer.db_to_schema(item)
        _log.info(f"Deleting {item}")

        await self._session.delete(item)
        await self._session.flush()
        return deleted_item
