import os
import zipfile
from typing import Any, Iterable

import yaml
from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy import update
from sqlalchemy.sql.expression import func

import common.models as models
import common.schemas as schemas
from admin_portal.config import JobsConfig
from common import utils
from common.config import multiline_logger as logger
from common.data import DataSourceConfig
from common.services import DataSourceSerializer, DataSourceService, DataSourceTypeService


class AdminPortalDataSourceService(DataSourceService):

    async def _parse_details_field(self, type_id: int, details: dict[str, Any]) -> DataSourceConfig:

        config_class = await DataSourceTypeService(self._session).get_config_class(type_id)

        try:
            parsed_config = config_class(**details)
        except ValidationError as e:
            logger.info(e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse 'details' field: {e}",
            )
        except Exception as e:
            logger.info(e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to parse 'details' field",
            )
        return parsed_config

    async def create_data_source(self, data: schemas.DataSourceBase) -> schemas.DataSource:
        parsed_config = await self._parse_details_field(data.type_id, data.details)

        item = models.DataSource(
            title=data.title,
            description=data.description,
            type_id=data.type_id,
            details=parsed_config.model_dump(mode='json', by_alias=True),
        )

        self._session.add(item)
        await self._session.commit()

        await self._session.refresh(item, attribute_names=["type"])
        return DataSourceSerializer.db_to_schema(item)

    @staticmethod
    async def export_data_sources(data_sources: Iterable[schemas.DataSource], res_dir: str) -> None:
        data = [
            source.model_dump(mode='json', include=JobsConfig.DATA_SOURCE_FIELDS)
            for source in data_sources
        ]
        data_sources_file = os.path.join(res_dir, JobsConfig.DATA_SOURCES_FILE)
        utils.write_yaml({'dataSources': data}, data_sources_file)

    async def import_data_sources_from_zip(
        self, zip_file: zipfile.ZipFile, update_data_sources: bool
    ) -> dict[str, schemas.DataSource]:
        existing_data_sources = {
            ds.title: ds for ds in await self.get_data_sources_schemas(limit=None, offset=0)
        }

        with zip_file.open(JobsConfig.DATA_SOURCES_FILE) as file:
            data_sources_json = yaml.safe_load(file)

        data_sources = {}
        for ds in data_sources_json['dataSources']:
            data_source_data = schemas.DataSourceBase.model_validate(ds)
            logger.info(f"Importing data source: {data_source_data!r}")

            if data_source := existing_data_sources.get(data_source_data.title):
                if update_data_sources:
                    data = {
                        field: getattr(data_source_data, field)
                        for field in schemas.DataSourceUpdate.model_fields.keys()
                        if getattr(data_source_data, field) != getattr(data_source, field)
                    }
                    if data:
                        logger.info(f"Updating data source '{data_source_data.title}' with {data}")
                        data_source = await self.update(
                            data_source.id, schemas.DataSourceUpdate(**data)
                        )
                    else:
                        logger.info(
                            f"Data source '{data_source_data.title}' exists and up-to-date."
                        )
                else:
                    logger.info(f"Data source '{data_source_data.title}' already exists. Skipping.")
            else:
                data_source = await self.create_data_source(data_source_data)
            data_sources[data_source.title] = data_source

        return data_sources

    async def update(self, item_id: int, data: schemas.DataSourceUpdate) -> schemas.DataSource:

        item = await self._get_item_or_raise(item_id)

        if data.details:
            parsed_config = await self._parse_details_field(item.type_id, data.details)
            data.details = parsed_config.model_dump(mode='json', by_alias=True)

        query = (
            update(models.DataSource)
            .where(models.DataSource.id == item.id)
            .values(**data.model_dump(exclude_unset=True), updated_at=func.now())
            .returning(models.DataSource)
        )
        item = (await self._session.execute(query)).scalar_one()
        await self._session.commit()

        await self._session.refresh(item, attribute_names=["type"])
        return DataSourceSerializer.db_to_schema(item)

    async def delete(self, item_id: int) -> None:
        item = await self._get_item_or_raise(item_id)
        logger.info(f"Deleting {item}")

        await self._session.delete(item)
        await self._session.commit()
