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
from statgpt.common.data.statgpt_sdmx_proxy.config import StatGptSdmxProxyDataSourceConfig
from statgpt.common.data.statgpt_sdmx_proxy.config_client import (
    ProxyConfigServerError,
    ProxyConfigValidationError,
    fetch_proxy_config,
    push_proxy_config,
)
from statgpt.common.schemas import AuditActionType, AuditEntityType
from statgpt.common.services import DataSourceSerializer, DataSourceService, DataSourceTypeService

_log = logging.getLogger(__name__)

# Serialization alias of `StatGptSdmxProxyDataSourceConfig.proxy_config`: `details` is stored
# camelCase, so this is the key the admin portal sees and sends back.
_PROXY_CONFIG_KEY = "proxyConfig"


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
    def _proxy_config_url(item: schemas.DataSource) -> str | None:
        """Return the resolved config server URL, or None if this is not an SDMX proxy source."""

        config_class = DataManager.get_config_class(item.type.name)
        if not issubclass(config_class, StatGptSdmxProxyDataSourceConfig):
            return None

        try:
            config = config_class.model_validate(item.details)
        except ValidationError as e:
            _log.warning(
                "Cannot resolve the proxy config server URL of data source %r: %s", item.title, e
            )
            return None
        return config.get_config_url()

    @staticmethod
    async def _fetch_proxy_config(url: str) -> tuple[bool, dict[str, Any] | None]:
        """Load a proxy configuration, reporting failures instead of raising.

        Returns:
            A `(fetched, config)` pair. `fetched` is False when the config server could not be
            read, which is distinct from the server answering that it holds no configuration yet.
        """

        try:
            return True, await fetch_proxy_config(url)
        except ProxyConfigServerError as e:
            _log.warning("Could not load the SDMX proxy configuration from %s: %s", url, e)
            return False, None

    async def _enrich_with_proxy_config(self, items: Sequence[schemas.DataSource]) -> None:
        """Populate `details.proxyConfig` of SDMX proxy data sources from their config server.

        The config server owns the value, so it is always read live instead of being cached or
        persisted. Data sources sharing a config server are fetched once. When a config server
        cannot be read the key is left absent - a data source must stay readable when its config
        server is down, and an absent key tells the admin portal not to offer the configuration
        for editing.
        """

        targets = [(item, url) for item in items if (url := self._proxy_config_url(item))]
        if not targets:
            return

        urls = list({url for _, url in targets})
        results = dict(
            zip(urls, await asyncio.gather(*(self._fetch_proxy_config(url) for url in urls)))
        )

        for item, url in targets:
            fetched, config = results[url]
            if fetched:
                item.details[_PROXY_CONFIG_KEY] = config

    @staticmethod
    async def _push_proxy_config(config: DataSourceConfig) -> None:
        """Send the submitted proxy configuration to the config server, if there is one.

        An absent configuration means "leave the config server alone", so a data source can be
        updated while its config server is unreachable without wiping the stored configuration.
        """

        if not isinstance(config, StatGptSdmxProxyDataSourceConfig):
            return
        if config.proxy_config is None:
            return

        url = config.get_config_url()
        try:
            await push_proxy_config(url, config.proxy_config)
        except ProxyConfigValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
            ) from e
        except ProxyConfigServerError as e:
            _log.warning("Could not update the SDMX proxy configuration at %s: %s", url, e)

    async def get_schema_by_id_with_proxy_config(self, item_id: int) -> schemas.DataSource:
        item = await self.get_schema_by_id(item_id)
        await self._enrich_with_proxy_config([item])
        return item

    async def get_data_sources_schemas_with_proxy_config(
        self, limit: int | None, offset: int
    ) -> list[schemas.DataSource]:
        items = await self.get_data_sources_schemas(limit=limit, offset=offset)
        await self._enrich_with_proxy_config(items)
        return items

    @audit_action(entity_type=AuditEntityType.DATA_SOURCE, action_type=AuditActionType.CREATE)
    async def create_data_source(self, data: schemas.DataSourceBase) -> schemas.DataSource:
        parsed_config = await self._parse_details_field(data.type_id, data.details)
        await self._push_proxy_config(parsed_config)

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
        await self._enrich_with_proxy_config([result])
        return result

    async def export_data_sources(
        self, data_sources: Iterable[schemas.DataSource], res_dir: str
    ) -> None:
        sources = list(data_sources)
        await self._enrich_with_proxy_config(sources)

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
            for ds in await self.get_data_sources_schemas_with_proxy_config(limit=None, offset=0)
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

        if data.details:
            parsed_config = await self._parse_details_field(item.type_id, data.details)
            await self._push_proxy_config(parsed_config)
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
        await self._enrich_with_proxy_config([result])
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
