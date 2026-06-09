import asyncio
import collections
import json
import logging
import os
import tempfile
import time
import typing as t
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime
from functools import cached_property, partial

import pandas as pd
import plotly.graph_objects as go
import sdmx.model.common
from dateutil.parser import parse
from sdmx.message import DataMessage, StructureMessage
from sdmx.model.common import Code
from sdmx.model.v21 import DataflowDefinition as DataFlow
from sdmx.model.v21 import DataStructureDefinition, TimeDimension

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import (
    CategoricalDimension,
    DataResponse,
    DataResponseStatus,
    DataSet,
    DataSetAvailabilityQuery,
    DataSetQuery,
    Dimension,
    DimensionDataType,
    DimensionQuery,
    DimensionType,
    OfflineDataSet,
    Query,
    QueryOperator,
    VirtualDimension,
    VirtualDimensionCategory,
)
from statgpt.common.data.sdmx.common import (
    BaseNameableArtefact,
    CodeCategory,
    CodeIndicator,
    ComplexIndicator,
    DataExplorerUrlConfig,
    SdmxCodeListDimension,
    SdmxDataSetAvailabilityQuery,
    SdmxDataSetConfig,
    SdmxDataSetQuery,
    SdmxDimension,
    SdmxQueryReadinessStatus,
    TimeDimensionQuery,
    UrnReference,
    build_data_explorer_dataset_url,
    build_data_explorer_url_query,
)
from statgpt.common.data.sdmx.python_code import generate_python_query_body
from statgpt.common.schemas.dataset import Status
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus
from statgpt.common.schemas.query import (
    JsonComponentQuery,
    JsonQueryMetadata,
    JsonQueryOperator,
    JsonQueryWithMetadata,
)
from statgpt.common.settings.sdmx import sdmx_settings
from statgpt.common.utils import escape_invalid_filename_chars
from statgpt.common.utils.plotly import PlotlyGraphBuilder, df_2_plotly_grid
from statgpt.common.utils.time_utils import get_time_period_bounds

from .attribute import Sdmx21Attribute, Sdmx21CodeListAttribute
from .schemas import Urn
from .utils import convert_keys_to_str

if t.TYPE_CHECKING:
    from statgpt.common.data.sdmx.v21.datasource import Sdmx21DataSourceHandler


_log = logging.getLogger(__name__)


class SdmxOfflineDataSet(OfflineDataSet[SdmxDataSetConfig, 'Sdmx21DataSourceHandler']):
    @property
    def source_id(self) -> str:
        return self.config.urn.short_urn()

    @property
    def description(self) -> str:
        return ''

    def get_resolved_sdmx1_source(self) -> str | None:
        return (
            self.config.sdmx1_source
            or self._datasource.config.sdmx1_source
            or self.config.urn.agency_id
        )


class InvalidConfigurationError(Exception):
    def __init__(self, message: str, entity_id: uuid.UUID, dataset_urn: str):
        super().__init__(message)
        self.entity_id = entity_id
        self.dataset_urn = dataset_urn

    def to_dict(self) -> dict[str, str]:
        return {
            "entity_id": str(self.entity_id),
            "dataset_urn": self.dataset_urn,
            "error": str(self),
        }


def _series_count_from_data_message(data_msg: DataMessage) -> int:
    """Number of series in the SDMX data message (per dataset, ``DataSet.series``)."""
    if not data_msg.data:
        return 0
    return sum(len(ds.series) for ds in data_msg.data)


class Sdmx21DataResponse(DataResponse):

    def __init__(
        self,
        dataset: 'Sdmx21DataSet',
        sdmx_query: SdmxDataSetQuery,
        df: pd.DataFrame,
        url: str | None,
        status: DataResponseStatus,
        display_series_count: int = 0,
        last_updated_at: datetime | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.sdmx_query = sdmx_query
        self.df = df
        self._url = url
        self._status = status
        self._display_series_count = display_series_count
        self._last_updated_at = last_updated_at

    @property
    def status(self) -> DataResponseStatus:
        return self._status

    @cached_property
    def file_name(self) -> str:
        return self.dataset.get_file_name()

    @cached_property
    def dataset_name(self) -> str:
        return f"{self.dataset.name} [{self.dataset.source_id}]"

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        return self._enrich_df_with_names(self.df)

    @cached_property
    def visual_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame suitable for visualization and export (Plotly grid, CSV file)"""
        if self.df.empty:
            return self.df
        time_dimension = self.dataset.get_time_dimension()
        visual_df: pd.DataFrame = self.df.unstack(time_dimension.entity_id)  # type: ignore[assignment]
        visual_df.columns = visual_df.columns.droplevel(0)
        visual_df = self._enrich_df_with_names(visual_df)
        return visual_df

    @cached_property
    def csv_dataframe(self) -> pd.DataFrame:
        """Return full SDMX observation-level dataframe for CSV export."""
        return self.dataframe

    @cached_property
    def dimension_ids(self) -> set[str]:
        return {d.entity_id for d in self.dataset.dimensions()}

    @cached_property
    def attribute_ids(self) -> set[str]:
        return {a.entity_id for a in self.dataset.attributes()}

    def get_display_series_count(self) -> int:
        """Number of series in the data message (set when the query is executed)."""
        return self._display_series_count

    def enrich_attachment_name(self, value: str) -> str:
        """Replace placeholders in the attachment name with actual values."""
        return value.format(
            dataset_source_id=self.dataset.source_id,
            dataset_name=self.dataset.name,
        )

    @property
    def resource_path(self) -> str:
        return self.dataset.source_id

    def merge(self, other: "DataResponse") -> "Sdmx21DataResponse":
        if not isinstance(other, Sdmx21DataResponse):
            raise TypeError(f"Cannot merge {type(other)} with {type(self)}")

        if self.dataset.entity_id != other.dataset.entity_id:
            raise ValueError(
                f"Cannot merge different datasets: {self.dataset.entity_id} != {other.dataset.entity_id}"
            )

        return Sdmx21DataResponse(
            dataset=self.dataset,
            df=pd.concat([self.df, other.df]).drop_duplicates(),
            sdmx_query=self.sdmx_query.merge(other.sdmx_query),
            url=None,
            status=self.status.merge(other.status),
            display_series_count=self._display_series_count + other._display_series_count,
            last_updated_at=self._last_updated_at,
        )

    @property
    def custom_table_dict(self) -> dict | None:
        if self.df.empty:
            return None

        data_json = json.loads(self.dataframe.to_json(orient='table'))

        series_count = self.visual_dataframe.shape[0]
        height = min(400, 75 + 27 * series_count)

        time_dimension = self.dataset.get_time_dimension()

        result = {
            'data': data_json,
            'metadata': {
                'time_column': time_dimension.entity_id,
                'pinned_columns': self.dataset.get_pinned_columns(),
            },
            'layout': {'height': height},
        }
        return result

    @property
    def plotly_grid(self) -> go.Figure | None:
        if self.visual_dataframe.empty:
            return None
        figure = df_2_plotly_grid(self.visual_dataframe, round_digits=2)
        return figure

    def get_plotly_graphs_with_names(self, template: str) -> list[tuple[str, go.Figure]]:
        try:
            graph_builder = PlotlyGraphBuilder(self.dataframe, self.dataset)
            graphs = graph_builder.plot_for_all_indicators()
            return [(self._graph_name(figure, template), figure) for figure in graphs]
        except Exception:
            _log.exception(f"Failed to create plots for dataset {self.dataset.short_urn}")
            return []

    @property
    def url_query(self) -> str | None:
        return self._url

    @property
    def json_query(self) -> JsonQueryWithMetadata:
        return JsonQueryWithMetadata(
            urn=self.dataset.short_urn,
            filters=self._to_component_filters(self.sdmx_query),
            metadata=JsonQueryMetadata(
                country_dimension=self.dataset.config.country_dimension,
                indicator_dimensions=self.dataset.config.indicator_dimensions,
                time_period_dimension=self.dataset.config.time_period_dimension_id,
                dataset_url=self.dataset.dataset_url,
                key_dimension_ids_in_dsd_order=self.dataset.sdmx_key_dimension_ids_in_dsd_order,
            ),
            sdmx1_source=self.dataset.get_resolved_sdmx1_source(),
            last_updated_at=self._last_updated_at,
        )

    def get_python_code_body(self, suffix: str = "") -> str | None:
        return self.dataset.get_python_code_body(self.sdmx_query, suffix=suffix)

    @cached_property
    def time_period(self) -> tuple[str, str] | None:
        """Return the time period covered by the data in this response as a tuple of (start, end)."""
        if self.df.empty:
            return None

        time_dimension = self.dataset.get_time_dimension()
        time_column = time_dimension.entity_id

        if time_column not in self.df.index.names:
            return None

        time_values = self.df.index.get_level_values(time_column).dropna().unique()
        if len(time_values) == 0:
            return None

        time_values_list = [str(v) for v in time_values]
        return get_time_period_bounds(time_values_list)

    def _enrich_df_with_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=True)
        df = df.reset_index()
        sorted_columns = []
        for column in df.columns:
            id2name_mapping = self.dataset.map_component_values_id_2_name(
                value_ids=df[column].to_list(), component_id=column
            )
            if id2name_mapping is None:
                continue
            id2name_imputed = {id_: name or id_ for id_, name in id2name_mapping.items()}

            id_column = column  # ToDo: consider adding _ID suffix
            name_column = f"{column}_Name"
            df[name_column] = df[column].map(id2name_imputed, na_action='ignore')
            for c in (id_column, name_column):
                sorted_columns.append(c)
        # append columns that were not enriched
        for column in df.columns:
            if column not in sorted_columns:
                sorted_columns.append(column)
        df = df[sorted_columns].copy()
        return df

    def _graph_name(self, figure: go.Figure, template: str) -> str:
        return template.format(
            dataset_source_id=self.dataset.source_id,
            dataset_name=self.dataset.name,
            figure_title=(figure.layout.title.text or '').replace('<br>', ' '),
        )

    @staticmethod
    def _create_time_dimension_query(
        time_dimension_query: TimeDimensionQuery | None,
    ) -> JsonComponentQuery | None:
        if not time_dimension_query:
            return None

        if time_dimension_query.start_period and time_dimension_query.end_period:
            return JsonComponentQuery(
                component_code=time_dimension_query.time_dimension_id,
                operator=JsonQueryOperator.BETWEEN,
                values=[
                    time_dimension_query.start_period,
                    time_dimension_query.end_period,
                ],
            )
        elif time_dimension_query.start_period:
            return JsonComponentQuery(
                component_code=time_dimension_query.time_dimension_id,
                operator=JsonQueryOperator.GE,
                values=[time_dimension_query.start_period],
            )
        elif time_dimension_query.end_period:
            return JsonComponentQuery(
                component_code=time_dimension_query.time_dimension_id,
                operator=JsonQueryOperator.LE,
                values=[time_dimension_query.end_period],
            )
        return None

    @classmethod
    def _to_component_filters(cls, sdmx_query: SdmxDataSetQuery) -> list[JsonComponentQuery]:
        res = [
            JsonComponentQuery(
                component_code=k,
                operator=JsonQueryOperator.IN,
                values=v,
            )
            for k, v in sdmx_query.categorical_dimensions.items()
        ]

        time_query = cls._create_time_dimension_query(sdmx_query.time_dimension_query)
        if time_query:
            res.append(time_query)

        return res


class Sdmx21DataSet(
    DataSet[SdmxDataSetConfig, 'Sdmx21DataSourceHandler'], BaseNameableArtefact[DataFlow]
):
    _dimensions: dict[str, SdmxDimension | VirtualDimension]
    _attributes: dict[str, Sdmx21Attribute]
    _virtual_dimensions: dict[str, VirtualDimension]
    _indicator_dimensions: dict[str, SdmxCodeListDimension | VirtualDimension]
    _country_dimension: SdmxCodeListDimension | VirtualDimension | None
    _dim_values_id_2_name: dict[str, dict[str, str]] | None

    def __init__(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: SdmxDataSetConfig,
        handler: 'Sdmx21DataSourceHandler',
        dataflow: DataFlow,
        locale: str,
        dimensions: Iterable[SdmxDimension],
        attributes: Iterable[Sdmx21Attribute],
    ):
        BaseNameableArtefact.__init__(self, dataflow, locale)
        DataSet.__init__(self, entity_id, title, config, handler)

        self._dimensions = {dimension.entity_id: dimension for dimension in dimensions}
        self._indicator_dimensions = {}
        self._virtual_dimensions = {}
        self._attributes = {attribute.entity_id: attribute for attribute in attributes}
        self._dim_values_id_2_name = None
        self._country_dimension = None

        # Add virtual dimensions
        for dim_id, dim_conf in config.dimensions.items():
            if dim_conf.virtual is not None:
                dimension = VirtualDimension(dim_id, dim_conf.virtual, alias=dim_conf.alias)
                self._virtual_dimensions[dimension.entity_id] = dimension
                self._dimensions[dimension.entity_id] = dimension

        self._validate_config_and_dimensions(config, self._dimensions, entity_id)

        # Set indicator dimensions
        for dim_id, dim_conf in config.dimensions.items():
            if dim_conf.type == DimensionType.INDICATOR:
                indicator_dimension = self._dimensions[dim_id]
                if not isinstance(indicator_dimension, SdmxCodeListDimension) and not isinstance(
                    indicator_dimension, VirtualDimension
                ):
                    raise InvalidConfigurationError(
                        f"Indicator dimension must be code list dimension or virtual dimension: {indicator_dimension}",
                        entity_id=entity_id,
                        dataset_urn=config.urn.short_urn(),
                    )
                self._indicator_dimensions[dim_id] = indicator_dimension

        if country_dimension_id := config.country_dimension:
            country_dimension = self._dimensions[country_dimension_id]
            if not isinstance(country_dimension, SdmxCodeListDimension | VirtualDimension):
                raise InvalidConfigurationError(
                    f"Country dimension must be code list dimension or virtual dimension: {country_dimension}",
                    entity_id=entity_id,
                    dataset_urn=config.urn.short_urn(),
                )
            self._country_dimension = country_dimension
        else:
            self._country_dimension = None

    @staticmethod
    def _validate_config_and_dimensions(
        config: SdmxDataSetConfig,
        dimensions: dict[str, SdmxDimension | VirtualDimension],
        entity_id: uuid.UUID,
    ) -> None:
        """Validate that the dataset is properly configured and raises `InvalidConfigurationError` if not."""

        not_found_dimensions = [
            dim_id for dim_id in config.dimensions.keys() if dim_id not in dimensions
        ]
        not_configured_dimensions = [
            dim_id for dim_id in dimensions.keys() if dim_id not in config.dimensions
        ]

        errors = []
        if not_found_dimensions:
            errors.append(
                f"Dimensions is configured but not found in the dataset: {not_found_dimensions}."
            )
        if not_configured_dimensions:
            errors.append(
                f"Dimensions is found in the dataset but not configured: {not_configured_dimensions}."
            )
        if errors:
            raise InvalidConfigurationError(
                " ".join(errors), entity_id=entity_id, dataset_urn=config.urn.short_urn()
            )

        return None

    async def updated_at(self, auth_context: AuthContext) -> datetime | None:
        if self.config.citation:
            if self.config.citation.last_updated:
                return parse(self.config.citation.last_updated)
        return None

    def get_resolved_config(self) -> dict:
        """
        Return config with resolved URN from the loaded dataflow.

        Handles dynamic URN values per SDMX standard:
        - version: "latest", "~" (latest stable), version wildcards
        - agency_id: "all" wildcard
        - resource_id: wildcards
        """
        actual_urn = Urn.for_artifact(self._artefact)

        if (
            actual_urn.agency_id == self._config.urn.agency_id
            and actual_urn.resource_id == self._config.urn.resource_id
            and actual_urn.version == self._config.urn.version
        ):
            return self._config.model_dump(mode='json', by_alias=True)

        resolved_urn = UrnReference(
            agency_id=actual_urn.agency_id,
            resource_id=actual_urn.resource_id,
            version=actual_urn.version,
        )
        resolved_config = self._config.model_copy(update={'urn': resolved_urn})
        return resolved_config.model_dump(mode='json', by_alias=True)

    @property
    def status(self) -> Status:
        return Status(status='online')

    @property
    def source_id(self) -> str:
        return self.short_urn

    @property
    def name(self) -> str:
        if self.config.use_title_from_src:
            return BaseNameableArtefact.name.fget(self)  # type: ignore
        else:
            return DataSet.name.fget(self)  # type: ignore

    @property
    def default_value_codes(self) -> list[str]:
        if self.config.default_value_codes is not None:
            # dataset-level override
            return self.config.default_value_codes
        return self._datasource.config.default_value_codes

    @property
    def dataset_url(self) -> str | None:
        if self._datasource.config.use_data_explorer_for_dataset_url:
            if base := self._resolved_data_explorer_base_url():
                return build_data_explorer_dataset_url(
                    base, self._short_urn, self._resolved_data_explorer_url_config()
                )
            _log.warning(
                "Data explorer URL is not configured on the dataset or data source: %s",
                self._datasource.source_id,
            )
        if self.config.citation and self.config.citation.url:
            return self.config.citation.get_url()
        return None

    def _resolved_data_explorer_base_url(self) -> str | None:
        if url := self.config.get_data_explorer_url():
            return url
        return self._datasource.config.get_data_explorer_url()

    def _resolved_data_explorer_url_config(self) -> DataExplorerUrlConfig | None:
        if self.config.data_explorer_url_config is not None:
            return self.config.data_explorer_url_config
        return self._datasource.config.data_explorer_url_config

    def _format_data_explorer_url(
        self,
        base_url: str,
        dsd: DataStructureDefinition,
        key_dict: dict[str, list[str]],
        sdmx_query: SdmxDataSetQuery,
    ) -> str:
        return build_data_explorer_url_query(
            base_url=base_url,
            short_urn=self._short_urn,
            key_dict=key_dict,
            sdmx_query=sdmx_query,
            config=self._resolved_data_explorer_url_config(),
            sdmx_key_builder=partial(convert_keys_to_str, dsd),
            dimension_values_resolver=self._resolve_explorer_dimension_values,
        )

    def _resolve_explorer_dimension_values(self, dimension_id: str, codes: list[str]) -> list[str]:
        id2name_mapping = self.map_component_values_id_2_name(codes, dimension_id)
        if id2name_mapping is None:
            return codes
        return [id2name_mapping.get(code) or code for code in codes]

    async def _get_available_series(
        self,
        cur_query: dict[str, str],
        cur_dim: str,
        cur_dim_avail_values: list[str],
        other_dims_to_fill: list[str],
        queries_count: int,
        auth_context: AuthContext,
    ) -> tuple[list[dict[str, str]], int]:
        series: list[dict[str, str]] = []

        new_query: dict[str, str]

        if len(other_dims_to_fill) == 0:
            for candidate in cur_dim_avail_values:
                new_query = cur_query.copy()
                new_query[cur_dim] = candidate
                series.append(new_query)
            return series, queries_count

        for candidate in cur_dim_avail_values:
            new_query = cur_query.copy()
            new_query[cur_dim] = candidate

            avail_query_raw = {
                k: Query(values=[v], operator=QueryOperator.IN) for k, v in new_query.items()
            }
            avail_query = DataSetAvailabilityQuery(dimensions_queries_dict=avail_query_raw)
            avail_query_resp = await self.availability_query(
                query=avail_query, auth_context=auth_context
            )
            queries_count += 1

            # select next dims to fill

            other_dims_set = set(other_dims_to_fill)
            other_dims_avail_values = {
                dim: avail_query_resp.dimensions_queries_dict[dim].values for dim in other_dims_set
            }

            # NOTE: process dimensions with a single candidate
            for dim, dim_avail_values in other_dims_avail_values.items():
                if len(dim_avail_values) == 1:
                    new_query[dim] = dim_avail_values[0]
                    other_dims_set.remove(dim)

            # check if there are any candidates left
            if not other_dims_set:
                series.append(new_query)
                continue

            # now we need to sort next dims
            other_dims_avail_values_cnt = {
                k: len(v)
                for k, v in other_dims_avail_values.items()
                # NOTE: it's also important to filter dimensions with a single candidate
                if k in other_dims_set
            }
            items_ordered = sorted(other_dims_avail_values_cnt.items(), key=lambda x: x[1])
            other_dims_ordered = [x[0] for x in items_ordered]

            # call recursively

            next_dim = other_dims_ordered[0]
            next_other_dims_to_fill = other_dims_ordered[1:]
            next_dim_avail_values = other_dims_avail_values[next_dim]

            call_res = await self._get_available_series(
                cur_query=new_query,
                cur_dim=next_dim,
                cur_dim_avail_values=next_dim_avail_values,
                other_dims_to_fill=next_other_dims_to_fill,
                queries_count=queries_count,
                auth_context=auth_context,
            )
            cur_series, queries_count = call_res
            series.extend(cur_series)

        return series, queries_count

    def _save_indicator_combinations(
        self,
        file_path: str,
        series: list[dict[str, str]],
        order: list[str],
        virtual_indicator_dimensions: Sequence[VirtualDimension],
    ) -> int:
        series_df = pd.DataFrame(series)
        series_df.sort_values(order, inplace=True)

        for dim in virtual_indicator_dimensions:
            series_df[dim.entity_id] = dim.value.entity_id

        series_df.to_csv(file_path, index=False)
        _log.info(f"{self.source_id}. Saved indicator combinations to '{file_path}'")
        duplicates = series_df.duplicated().sum()

        return duplicates

    async def _load_indicator_combinations_to(
        self, file_path: str, auth_context: AuthContext
    ) -> None:
        indicator_ids = [x.entity_id for x in self.indicator_dimensions(non_virtual=True)]
        _log.info(
            f'{self.source_id}. Will extract available combinations for following indicator dimensions: {indicator_ids}'
        )

        time_start = time.monotonic()

        query = DataSetAvailabilityQuery(dimensions_queries_dict={})
        avail_query_resp = await self.availability_query(query=query, auth_context=auth_context)

        avail_values = {
            dim: avail_query_resp.dimensions_queries_dict[dim].values for dim in indicator_ids
        }

        dim_2_avail_values_cnt_sorted = sorted(
            ((k, len(v)) for k, v in avail_values.items()), key=lambda x: x[1]
        )
        _log.info(f"{self.source_id} {dim_2_avail_values_cnt_sorted=}")
        order = [x[0] for x in dim_2_avail_values_cnt_sorted]

        series, queries_count = await self._get_available_series(
            cur_query={},
            cur_dim=order[0],
            cur_dim_avail_values=avail_values[order[0]],
            other_dims_to_fill=order[1:],
            queries_count=1,
            auth_context=auth_context,
        )
        _log.info(f'{self.source_id}. Extracted available indicator dimension combinations')
        _log.info(
            f"{self.source_id}. Number of series extracted: {len(series)}. Number of queries sent: {queries_count}"
        )

        elapsed_time = time.monotonic() - time_start
        _log.info(f'{self.source_id}. elapsed time: {elapsed_time :.3f} sec')

        virtual_indicator_dimensions = self.virtual_indicator_dimensions()

        duplicates = await asyncio.to_thread(
            self._save_indicator_combinations,
            file_path,
            series,
            order,
            virtual_indicator_dimensions,
        )
        if duplicates:
            raise ValueError(
                f"{self.source_id}. Found {duplicates} duplicates in the indicator combinations"
            )
        _log.info(f"{self.source_id}. No duplicates found in the indicator combinations")

    def _read_indicator_combinations_from_file(self, file_path: str) -> pd.DataFrame:
        _log.debug(f"{self.source_id}. Reading indicator combinations from '{file_path}'")
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        _log.debug(f"{self.source_id}. Read {len(df)} indicator combinations from '{file_path}'")
        return df

    async def _get_or_load_indicator_combinations(
        self, auth_context: AuthContext, allow_cached: bool
    ) -> pd.DataFrame:
        file_name = escape_invalid_filename_chars(f"{self.source_id}.csv")

        if cache_dir := sdmx_settings.cache_dir:
            dir_name = os.path.join(cache_dir, sdmx_settings.indicator_combinations_subdir)
            file_path = os.path.join(str(dir_name), file_name)

            if not allow_cached or not os.path.exists(file_path):
                status = 'not found' if allow_cached else 'not allowed'
                _log.info(f"{self.source_id}. Indicator combinations cache {status}.")
                # Create cache of available indicator combinations:
                os.makedirs(dir_name, exist_ok=True)
                await self._load_indicator_combinations_to(file_path, auth_context=auth_context)
            else:
                _log.info(f"{self.source_id}. Getting indicator combinations from cache.")

            return await asyncio.to_thread(self._read_indicator_combinations_from_file, file_path)
        else:
            _log.info(
                f"{self.source_id}. Indicator combinations cache disabled. Loading to temp dir..."
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_path = os.path.join(tmp_dir, file_name)
                await self._load_indicator_combinations_to(file_path, auth_context)
                return await asyncio.to_thread(
                    self._read_indicator_combinations_from_file, file_path
                )

    async def _indicators_from_dimensions(
        self, auth_context: AuthContext, allow_cached: bool
    ) -> list[ComplexIndicator]:

        df_avail_dim_combinations = await self._get_or_load_indicator_combinations(
            auth_context=auth_context, allow_cached=allow_cached
        )

        # Use the same columns order as in the list of indicator dimensions from dataset config.
        df_avail_dim_combinations = df_avail_dim_combinations[self._indicator_dimensions.keys()]

        dim_cat_id_2_model: dict[str, dict[str, CodeCategory | VirtualDimensionCategory]] = (
            collections.defaultdict(dict)
        )
        for dimension in self._indicator_dimensions.values():
            if isinstance(dimension, SdmxCodeListDimension):
                for code_category in dimension.available_values:
                    dim_cat_id_2_model[dimension.entity_id][code_category.query_id] = code_category
            elif isinstance(dimension, VirtualDimension):
                dim_cat_id_2_model[dimension.entity_id][dimension.value.entity_id] = dimension.value

        def _create_indicator(row):
            return ComplexIndicator(
                CodeIndicator(dim_cat_id_2_model[dim_id][row[dim_id]])
                for dim_id in df_avail_dim_combinations.columns
            )

        df_indicators = await asyncio.to_thread(
            df_avail_dim_combinations.apply, _create_indicator, axis=1  # type: ignore
        )
        indicators = df_indicators.to_list()
        return indicators

    @staticmethod
    def _operator_to_str(operator: QueryOperator) -> str:
        if operator == QueryOperator.EQUALS or operator == QueryOperator.IN:
            return "eq"
        elif operator == QueryOperator.NOT_EQUALS:
            return "ne"
        elif operator == QueryOperator.GREATER_THAN:
            return "gt"
        elif operator == QueryOperator.LESS_THAN:
            return "lt"
        elif operator == QueryOperator.GREATER_THAN_OR_EQUALS:
            return "ge"
        elif operator == QueryOperator.LESS_THAN_OR_EQUALS:
            return "le"
        else:
            raise ValueError(f"Operator {operator} is not supported")

    def get_time_dimension(self) -> SdmxDimension:
        for dimension in self._dimensions.values():
            if isinstance(dimension, SdmxDimension) and dimension.is_time_dimension:
                return dimension
        raise ValueError("No time dimension")

    def get_frequency_dimension(self) -> VirtualDimension | SdmxDimension:
        if self.config.frequency_dimension not in self._dimensions:
            raise ValueError(f"No frequency dimension (id={self.config.frequency_dimension!r})")
        return self._dimensions[self.config.frequency_dimension]

    @staticmethod
    def _append_category_query(
        query: DimensionQuery,
        dimension: SdmxDimension,
        result: SdmxDataSetQuery | SdmxDataSetAvailabilityQuery,
    ):
        if query.is_all_selected:
            # if all values are requested, we do not append anything
            return
        result.categorical_dimensions[dimension.entity_id] = query.values

    @staticmethod
    def _append_time_dimension_query(
        query: DimensionQuery, result: SdmxDataSetQuery | SdmxDataSetAvailabilityQuery
    ):
        values = query.values  # expecting values in format YYYY-MM-DD
        if query.operator == QueryOperator.BETWEEN:
            result.time_dimension_query = TimeDimensionQuery(
                time_dimension_id=query.dimension_id,
                start_period=values[0],
                end_period=values[1],
            )
        elif query.operator == QueryOperator.EQUALS:
            value = values[0]
            result.time_dimension_query = TimeDimensionQuery(
                time_dimension_id=query.dimension_id, start_period=value, end_period=value
            )
        elif query.operator == QueryOperator.GREATER_THAN_OR_EQUALS:
            result.time_dimension_query = TimeDimensionQuery(
                time_dimension_id=query.dimension_id, start_period=values[0], end_period=None
            )  # type: ignore
        elif query.operator == QueryOperator.LESS_THAN_OR_EQUALS:
            result.time_dimension_query = TimeDimensionQuery(
                time_dimension_id=query.dimension_id, start_period=None, end_period=values[0]
            )  # type: ignore
        else:
            raise ValueError(f"Unsupported operator {query.operator}")

    def _evaluate_data_query_status(
        self, data_query: DataSetQuery
    ) -> tuple[SdmxQueryReadinessStatus, list[str]]:
        status = SdmxQueryReadinessStatus.READY
        missing_dimensions = []
        dimension_queries = data_query.dimensions_queries_dict
        for dimension_id, dimension in self._dimensions.items():
            if (
                isinstance(dimension, SdmxDimension)
                and dimension.is_time_dimension
                and dimension_id not in dimension_queries
            ):
                missing_dimensions.append(dimension_id)
                status = SdmxQueryReadinessStatus.MISSING_REQUIRED_DIMENSIONS
            elif dimension.is_mandatory and dimension_id not in dimension_queries:
                missing_dimensions.append(dimension_id)
                status = SdmxQueryReadinessStatus.MISSING_REQUIRED_DIMENSIONS
        return status, missing_dimensions

    def _append_dimension_query(self, query: DimensionQuery, result: SdmxDataSetQuery):
        dimension = self._dimensions[query.dimension_id]
        if isinstance(dimension, VirtualDimension):
            # virtual dimensions are not used in queries
            return
        elif dimension.dimension_type == DimensionDataType.CATEGORY:
            self._append_category_query(query, dimension, result)
        elif dimension.is_time_dimension:
            self._append_time_dimension_query(query, result)

    def _to_sdmx_query(self, query: DataSetQuery) -> SdmxDataSetQuery:
        result = SdmxDataSetQuery.empty(query.uuid)
        # self._append_indicator_query_to_dataset_query(query, result)
        for dimension_query in query.dimensions_queries:
            if not dimension_query.values and dimension_query.operator != QueryOperator.ALL:
                continue
            dimension = self._dimensions[dimension_query.dimension_id]
            if isinstance(dimension, VirtualDimension):
                # virtual dimensions are not used in queries
                continue
            if dimension.dimension_type == DimensionDataType.CATEGORY:
                self._append_category_query(dimension_query, dimension, result)
        # appending datetime queries at the end, so we can use the value for frequency
        for dimension_query in query.dimensions_queries:
            dimension = self._dimensions[dimension_query.dimension_id]
            if isinstance(dimension, VirtualDimension):
                # virtual dimensions are not used in queries
                continue
            elif dimension.is_time_dimension:
                self._append_time_dimension_query(dimension_query, result)
        return result

    def _to_sdmx_availability_query(
        self, availability_query: DataSetAvailabilityQuery
    ) -> SdmxDataSetAvailabilityQuery:
        result = SdmxDataSetAvailabilityQuery()  # type: ignore

        # self._append_indicator_query_to_availability_query(availability_query, result)

        for dimension_query in availability_query.dimensions_queries:
            dimension = self._dimensions[dimension_query.dimension_id]
            if isinstance(dimension, VirtualDimension):
                # virtual dimensions are not used in queries
                continue

            if dimension.is_time_dimension:
                self._append_time_dimension_query(dimension_query, result)
            if not dimension_query.values:
                continue
            if dimension.dimension_type == DimensionDataType.CATEGORY:
                self._append_category_query(dimension_query, dimension, result)

        return result

    def code_list_dimensions(self) -> t.List[SdmxCodeListDimension]:
        return [
            dimension
            for dimension in self.dimensions()
            if isinstance(dimension, SdmxCodeListDimension)
        ]

    def dimensions(self) -> t.Sequence[SdmxDimension | VirtualDimension]:
        return list(self._dimensions.values())

    def dimension(self, dimension_id: str) -> SdmxDimension | VirtualDimension:
        return self._dimensions[dimension_id]

    def attributes(self) -> t.Sequence[Sdmx21Attribute]:
        return list(self._attributes.values())

    def non_indicator_dimensions(self) -> list[SdmxDimension | VirtualDimension]:
        return [
            self._dimensions[dim_id] for dim_id, dim_conf in self._config.non_indicator_dimensions
        ]

    def special_dimensions(self) -> dict[str, Dimension]:
        return {
            dim_conf.processor_id: self._dimensions[dim_id]
            for dim_id, dim_conf in self._config.special_dimensions
        }

    def indicator_dimensions(
        self, non_virtual: bool = False
    ) -> Sequence[SdmxCodeListDimension | VirtualDimension]:
        if non_virtual:
            return [
                dim
                for dim in self._indicator_dimensions.values()
                if not isinstance(dim, VirtualDimension)
            ]
        return list(self._indicator_dimensions.values())

    def virtual_indicator_dimensions(self) -> t.Sequence[VirtualDimension]:
        return [
            dim for dim in self._indicator_dimensions.values() if isinstance(dim, VirtualDimension)
        ]

    @cached_property
    def required_dimensions(self) -> list[str]:
        res = [
            dim_id for dim_id, dim_conf in self.config.dimensions.items() if dim_conf.is_required
        ]
        return res

    async def get_indicators(
        self, auth_context: AuthContext, allow_cached: bool
    ) -> Sequence[ComplexIndicator]:
        return await self._indicators_from_dimensions(
            auth_context=auth_context, allow_cached=allow_cached
        )

    def country_dimension(self) -> CategoricalDimension | None:
        return self._country_dimension

    def _get_dim_values_id_2_name_mapping(self) -> dict[str, dict[str, str]]:
        if self._dim_values_id_2_name is not None:
            return self._dim_values_id_2_name

        self._dim_values_id_2_name = {}
        for dim in self.dimensions():
            if not isinstance(dim, SdmxCodeListDimension):
                continue
            self._dim_values_id_2_name[dim.entity_id] = {
                code.query_id: code.name for code in dim.code_list.codes()
            }

        return self._dim_values_id_2_name

    def map_component_values_id_2_name(
        self, value_ids: Iterable[str], component_id: str
    ) -> dict[str, str | None] | None:
        """Map dimension or attribute ids to their corresponding names."""

        component: SdmxCodeListDimension | Sdmx21CodeListAttribute
        if component_id in self._dimensions:
            dimension = self._dimensions[component_id]
            if isinstance(dimension, VirtualDimension):
                value_ids_list = list(value_ids)
                if len(value_ids_list) != 1 or dimension.value.entity_id != value_ids_list[0]:
                    raise ValueError(
                        f"Inappropriate value ids for virtual dimension {component_id!r}: {value_ids_list!r}"
                    )
                return {dimension.value.entity_id: dimension.value.name}
            if not isinstance(dimension, SdmxCodeListDimension):
                _log.debug(
                    "Dimension %s of dataset %s is not a code list dimension",
                    component_id,
                    self.short_urn,
                )
                return None
            component = dimension
        elif component_id in self._attributes:
            attribute = self._attributes[component_id]
            if not isinstance(attribute, Sdmx21CodeListAttribute):
                _log.debug(
                    "Attribute %s of dataset %s is not a code list attribute",
                    component_id,
                    self.short_urn,
                )
                return None
            component = attribute
        else:
            _log.debug("Component %s not found in dataset %s", component_id, self.short_urn)
            return None

        code_list = component.code_list
        res = {
            value_id: code_list[value_id].name if value_id in code_list else None
            for value_id in value_ids
            if isinstance(value_id, str)
        }
        return res

    def map_dim_queries_2_names(
        self, queries: dict[str, list[str]]
    ) -> dict[str, dict[str, str | None]]:
        """
        queries: {dimension_id: [list of term_ids]}
        """
        res: dict[str, dict[str, str | None]] = {}
        for dim_id, term_ids in queries.items():
            id2name = self.map_component_values_id_2_name(term_ids, component_id=dim_id)
            if id2name is None:
                raise ValueError(f'Unexpected dimension id: "{dim_id}"')
            res[dim_id] = id2name
        return res

    def get_pinned_columns(self) -> list[str]:
        if self.config.pinned_columns:
            return self.config.pinned_columns
        pinned_columns = []
        if self.config.country_dimension:
            pinned_columns.append(f"{self.config.country_dimension}_Name")
        # limit dimensions to 3
        dimensions = (
            list(self.config.indicator_dimensions)[:3]
            if len(self.config.indicator_dimensions) > 3
            else self.config.indicator_dimensions
        )
        for dimension in dimensions:
            pinned_columns.append(f"{dimension}_Name")
        return pinned_columns

    def _availability_result_to_query(
        self, availability_result: StructureMessage
    ) -> DataSetAvailabilityQuery:

        constraints = list(availability_result.constraint.values())
        if len(constraints) == 0:
            return DataSetAvailabilityQuery()  # empty query
        elif len(constraints) != 1:
            raise ValueError("Unexpected quantity of constraints in structure message")
        constraint = constraints[0]
        if len(constraint.data_content_region) == 0:
            return DataSetAvailabilityQuery()  # empty query
        if len(constraint.data_content_region) != 1:
            raise ValueError("Unexpected quantity of cube-regions in constraint")
        cube_region = constraint.data_content_region[0]
        member_dict = cube_region.member
        dimension_to_available_values = {
            dim.id: {v.value for v in selection.values} for dim, selection in member_dict.items()
        }
        result = DataSetAvailabilityQuery()  # type: ignore
        for dimension_id, available_values in dimension_to_available_values.items():
            result.add_dimension_query(
                DimensionQuery(
                    dimension_id=dimension_id,
                    values=list(available_values),
                    operator=QueryOperator.IN,
                )
            )
        # append virtual dimensions
        for dimension_id, dimension in self._virtual_dimensions.items():
            result.add_dimension_query(
                DimensionQuery(
                    dimension_id=dimension_id,
                    values=[dimension._value.query_id],
                    operator=QueryOperator.IN,
                )
            )
        return result

    def _preview_result_to_query(
        self, series_keys: list[sdmx.model.SeriesKey]
    ) -> list[DataSetAvailabilityQuery]:
        results = []
        for series_key in series_keys:
            result = DataSetAvailabilityQuery()

            for key_value in series_key:  # sdmx.model.KeyValue
                result.add_dimension_query(
                    DimensionQuery(
                        dimension_id=key_value.id,
                        values=[key_value.value],
                        operator=QueryOperator.IN,
                    )
                )

            # append virtual dimensions
            for dimension_id, dimension in self._virtual_dimensions.items():
                result.add_dimension_query(
                    DimensionQuery(
                        dimension_id=dimension_id,
                        values=[dimension._value.query_id],
                        operator=QueryOperator.IN,
                    )
                )
            results.append(result)

        return results

    async def availability_query(
        self, query: DataSetAvailabilityQuery, auth_context: AuthContext
    ) -> DataSetAvailabilityQuery:
        sdmx_availability_query = self._to_sdmx_availability_query(query)
        client = await self._datasource.create_sdmx_client(auth_context)
        availability_result = await client.availableconstraint(
            agency_id=self._artefact.maintainer.id,  # type: ignore
            resource_id=self._artefact.id,
            version=self._artefact.version,  # type: ignore
            dsd=self._artefact.structure,
            params=sdmx_availability_query.get_params() or None,
            key=sdmx_availability_query.get_key() or None,
            use_cache=False,
        )
        result = self._availability_result_to_query(availability_result)
        return result

    def _data_msg_to_dataframe_sync(self, data_msg: DataMessage) -> pd.DataFrame:
        """Convert SDMX data message to Pandas DataFrame."""

        kwargs = {}
        if self.config.include_attributes:
            kwargs['attributes'] = "osgd"  # include observation, series, group, dataset attributes

        sdmx_pandas = sdmx.to_pandas(data_msg, **kwargs)
        if isinstance(sdmx_pandas, pd.Series):
            sdmx_pandas = sdmx_pandas.to_frame()
        elif not isinstance(sdmx_pandas, pd.DataFrame):
            raise ValueError(f"Got unexpected type from sdmx.to_pandas: {type(sdmx_pandas)}")

        return sdmx_pandas

    async def _data_msg_to_dataframe(self, data_msg: DataMessage) -> pd.DataFrame:
        return await asyncio.to_thread(self._data_msg_to_dataframe_sync, data_msg)

    def _include_attributes_sync(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not self.config.include_attributes:
            return df

        # Note: the data result might contain not all attributes, so we need to filter them
        attributes = [a for a in self.config.include_attributes if a in df.columns]

        # Remove unspecified attributes
        res_df = df[['value', *attributes]].copy()

        def _extract_attribute_value(x: t.Any) -> t.Any:
            """Extract attribute values (each cell initially contains the `AttributeValue` class or None)"""
            try:
                value = x.value
                if isinstance(value, Code):
                    return value.id
                return value
            except AttributeError:
                return x

        for attribute in attributes:
            res_df[attribute] = res_df[attribute].apply(_extract_attribute_value)

        # Add attributes to index
        res_df = res_df.set_index(keys=attributes, append=True)
        return res_df

    async def _include_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        return await asyncio.to_thread(self._include_attributes_sync, df)

    def _get_query_params(self, sdmx_query: SdmxDataSetQuery) -> dict:
        return sdmx_query.get_params_v21()

    async def _query_sdmx_data(
        self, sdmx_query: SdmxDataSetQuery, auth_context: AuthContext
    ) -> DataMessage:
        _log.info(f"Querying dataset '{self.source_id}' with {sdmx_query}")
        client = await self._datasource.create_sdmx_client(auth_context)
        data_msg: DataMessage = await client.data(
            agency_id=self._artefact.maintainer.id,  # type: ignore
            resource_id=self._artefact.id,
            version=self._artefact.version,  # type: ignore
            key=sdmx_query.get_key(),
            params=self._get_query_params(sdmx_query),
            dsd=self._artefact.structure,
        )
        return data_msg

    async def query(
        self, query: DataSetQuery, auth_context: AuthContext
    ) -> Sdmx21DataResponse | None:
        status, missing_dimensions = self._evaluate_data_query_status(query)
        if status != SdmxQueryReadinessStatus.READY:
            raise ValueError(
                f"Query is not ready: {query}, missing dimensions: {missing_dimensions}"
            )
        sdmx_query = self._to_sdmx_query(query)
        last_updated_at = await self.updated_at(auth_context)

        try:
            data_msg = await self._query_sdmx_data(sdmx_query, auth_context)
        except Exception as e:
            _log.exception(e)
            return Sdmx21DataResponse(
                dataset=self,
                df=pd.DataFrame(),
                sdmx_query=sdmx_query,
                url=None,
                status=DataResponseStatus(
                    request_status=DataRequestStatus.FAILED,
                    parsing_status=DataParsingStatus.NA,
                ),
                display_series_count=0,
                last_updated_at=last_updated_at,
            )

        if not data_msg:
            return Sdmx21DataResponse(
                dataset=self,
                df=pd.DataFrame(),
                sdmx_query=sdmx_query,
                url=None,
                status=DataResponseStatus(
                    request_status=DataRequestStatus.SUCCESS,
                    parsing_status=DataParsingStatus.FAILED,
                ),
                display_series_count=0,
                last_updated_at=last_updated_at,
            )

        explorer_cfg = self._resolved_data_explorer_url_config() or DataExplorerUrlConfig()
        if not explorer_cfg.view_in_data_explorer:
            url = None
        elif explorer_base := self._resolved_data_explorer_base_url():
            url = self._format_data_explorer_url(
                base_url=explorer_base,
                dsd=self._artefact.structure,
                key_dict=sdmx_query.get_key(),
                sdmx_query=sdmx_query,
            )
        else:
            http_response = data_msg.response
            url = http_response.url if http_response is not None else None

        message_series_count = _series_count_from_data_message(data_msg)
        try:
            sdmx_pandas = await self._data_msg_to_dataframe(data_msg)
            sdmx_pandas = await self._include_attributes(sdmx_pandas)
        except Exception as e:
            _log.exception(e)
            return Sdmx21DataResponse(
                dataset=self,
                df=pd.DataFrame(),
                sdmx_query=sdmx_query,
                url=url,
                status=DataResponseStatus(
                    request_status=DataRequestStatus.SUCCESS,
                    parsing_status=DataParsingStatus.FAILED,
                ),
                display_series_count=message_series_count,
                last_updated_at=last_updated_at,
            )

        return Sdmx21DataResponse(
            dataset=self,
            df=sdmx_pandas,
            sdmx_query=sdmx_query,
            url=url,
            status=DataResponseStatus(
                request_status=DataRequestStatus.SUCCESS,
                parsing_status=DataParsingStatus.SUCCESS,
            ),
            display_series_count=message_series_count,
            last_updated_at=last_updated_at,
        )

    def get_resolved_sdmx1_source(self) -> str | None:
        return (
            self._config.sdmx1_source
            or self._datasource.config.sdmx1_source
            or self._artefact.maintainer.id  # type: ignore[union-attr]
        )

    def get_python_code_body(self, sdmx_query: SdmxDataSetQuery, suffix: str = "") -> str:
        provider = self.get_resolved_sdmx1_source()
        if provider is None:
            raise RuntimeError("Cannot generate Python code without a resolved SDMX 1 source")

        flow_ref = (
            f"{self._artefact.maintainer.id}"  # type: ignore[union-attr]
            f",{self._artefact.id}"
            f",{self._artefact.version}"
        )
        key_string = self._dict_key_to_sdmx_string(sdmx_query.get_key())

        return generate_python_query_body(
            provider=provider,
            flow_ref=flow_ref,
            key=key_string,
            params=sdmx_query.get_params_v21(),
            suffix=suffix,
        )

    @cached_property
    def sdmx_key_dimension_ids_in_dsd_order(self) -> list[str]:
        """Non-time DSD dimension ids in REST key order."""
        return [
            dim.id
            for dim in self._artefact.structure.dimensions  # type: ignore[union-attr]
            if not isinstance(dim, TimeDimension)
        ]

    def _dict_key_to_sdmx_string(self, keys: dict[str, list[str]]) -> str:
        """Convert a dict key to an SDMX REST key string in DSD dimension order.

        The SDMX 2.1 REST API expects key as ordered dimension values
        separated by '.', with '+' joining multiple values within a dimension.
        Time dimensions are excluded (handled via query params).
        """
        parts = []
        for dim_id in self.sdmx_key_dimension_ids_in_dsd_order:
            values = keys.get(dim_id, [])
            parts.append("+".join(values))
        return ".".join(parts)
