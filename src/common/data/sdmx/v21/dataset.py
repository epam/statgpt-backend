import collections
import json
import os
import tempfile
import time
import typing as t
from collections.abc import Iterable
from datetime import datetime
from functools import cached_property

import pandas as pd
import plotly.graph_objects as go
import requests
import sdmx.model.common
from dateutil.parser import parse
from sdmx.message import DataMessage
from sdmx.model.common import Code
from sdmx.model.v21 import DataflowDefinition as DataFlow

from common.auth.auth_context import AuthContext
from common.config.logging import multiline_logger as logger
from common.data.base import (
    CategoricalDimension,
    DataResponse,
    DataSet,
    DataSetAvailabilityQuery,
    DataSetQuery,
    Dimension,
    DimensionQuery,
    DimensionType,
    OfflineDataSet,
    Query,
    QueryOperator,
    VirtualDimension,
)
from common.data.sdmx.common import (
    BaseNameableArtefact,
    CodeCategory,
    CodeIndicator,
    ComplexIndicator,
    FixedItem,
    SdmxCodeListDimension,
    SdmxDataSetConfig,
    SdmxDimension,
)
from common.schemas.dataset import Status
from common.settings.sdmx import sdmx_settings
from common.utils import escape_invalid_filename_chars
from common.utils.plotly import PlotlyGraphBuilder, df_2_plotly_grid

from .attribute import Sdmx21Attribute, Sdmx21CodeListAttribute
from .query import (
    JsonComponentQuery,
    JsonQueryMetadata,
    JsonQueryOperator,
    JsonQueryWithMetadata,
    SdmxDataSetAvailabilityQuery,
    SdmxDataSetQuery,
    SdmxQueryReadinessStatus,
    TimeDimensionQuery,
)

if t.TYPE_CHECKING:
    from common.data.sdmx.v21.datasource import Sdmx21DataSourceHandler


class SdmxOfflineDataSet(OfflineDataSet[SdmxDataSetConfig, 'Sdmx21DataSourceHandler']):
    @property
    def source_id(self) -> str:
        return self.config.urn

    @property
    def description(self) -> str:
        return ''


class Sdmx21DataResponse(DataResponse):

    def __init__(
        self,
        dataset: 'Sdmx21DataSet',
        sdmx_query: SdmxDataSetQuery,
        df: pd.DataFrame,
        url: str | None,
    ):
        self.dataset = dataset
        self.sdmx_query = sdmx_query
        self.df = df
        self._url = url

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
        time_dimension = self.dataset.get_time_dimension()
        visual_df: pd.DataFrame = self.df.unstack(time_dimension.entity_id)  # type: ignore[assignment]
        visual_df.columns = visual_df.columns.droplevel(0)
        visual_df = self._enrich_df_with_names(visual_df)
        return visual_df

    def enrich_attachment_name(self, value: str) -> str:
        """Replace placeholders in the attachment name with actual values."""
        return value.format(
            dataset_source_id=self.dataset.source_id,
            dataset_name=self.dataset.name,
        )

    def merge(self, other: "DataResponse") -> "Sdmx21DataResponse":
        if not isinstance(other, Sdmx21DataResponse):
            raise TypeError(f"Cannot merge {type(other)} with {type(self)}")

        if self.dataset.entity_id != other.dataset.entity_id:
            raise ValueError(
                f"Cannot merge different datasets: {self.dataset.entity_id} != {other.dataset.entity_id}"
            )

        return Sdmx21DataResponse(
            dataset=self.dataset,
            df=pd.concat([self.df, other.df]),
            sdmx_query=self.sdmx_query.merge(other.sdmx_query),
            url=None,
        )

    @property
    def custom_table_dict(self) -> dict | None:
        data_json = json.loads(self.dataframe.to_json(orient='table'))

        if self.visual_dataframe is not None:
            series_count = self.visual_dataframe.shape[0]
            height = min(400, 75 + 27 * series_count)
        else:
            height = 400

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
        if self.visual_dataframe is None:
            return None
        figure = df_2_plotly_grid(self.visual_dataframe, round_digits=2)
        return figure

    def get_plotly_graphs_with_names(self, template: str) -> list[tuple[str, go.Figure]]:
        try:
            graph_builder = PlotlyGraphBuilder(self.dataframe, self.dataset)
            graphs = graph_builder.plot_for_all_indicators()
            return [(self._graph_name(figure, template), figure) for figure in graphs]
        except Exception:
            logger.exception(f"Failed to create plots for dataset {self.dataset.short_urn}")
            return []

    @property
    def url_query(self) -> str | None:
        return self._url

    @property
    def json_query_old(self) -> dict:
        return {
            'urn': self.dataset.short_urn,
            'metadata': self._get_dataset_metadata_as_dict(),
            'filters': self._to_sdmx_filters(self.sdmx_query),
        }

    @property
    def json_query(self) -> dict:
        return JsonQueryWithMetadata(
            urn=self.dataset.short_urn,
            filters=self._to_component_filters(self.sdmx_query),
            metadata=JsonQueryMetadata(
                country_dimension=self.dataset.config.country_dimension,
                indicator_dimensions=self.dataset.config.indicator_dimensions,
            ),
        ).model_dump(by_alias=True)

    @property
    def python_code(self) -> str | None:
        return self.dataset.get_python_code(self.sdmx_query)

    def _enrich_df_with_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=True)
        df = df.reset_index()
        sorted_columns = []
        for column in df.columns:
            id2name_mapping = self.dataset.map_dim_values_id_2_name(
                value_ids=df[column].to_list(), dimension_name=column
            )
            if id2name_mapping is None:
                continue
            df[f"{column}_Name"] = df[column].map(id2name_mapping, na_action='ignore')
            sorted_columns.append(column)
            sorted_columns.append(f"{column}_Name")
        # append columns that were not enriched
        for column in df.columns:
            if column not in sorted_columns:
                sorted_columns.append(column)
        df = df[sorted_columns].copy()
        return df

    def _get_dataset_metadata_as_dict(self) -> dict[str, t.Any]:
        dataset_config = self.dataset.config
        return {
            'countryDimension': dataset_config.country_dimension,
            'indicatorDimensions': dataset_config.indicator_dimensions,
        }

    def _graph_name(self, figure: go.Figure, template: str) -> str:
        return template.format(
            dataset_source_id=self.dataset.source_id,
            dataset_name=self.dataset.name,
            figure_title=figure.layout.title.text.replace('<br>', ' '),
        )

    @staticmethod
    def _to_sdmx_filters(sdmx_query: SdmxDataSetQuery) -> list[dict[str, str]]:
        res = [
            {
                "componentCode": k,
                "operator": "in",
                "values": ','.join(v),
            }
            for k, v in sdmx_query.categorical_dimensions.items()
        ]

        if sdmx_query.time_dimension_query:
            res.append(
                {
                    "componentCode": sdmx_query.time_dimension_query.time_dimension_id,
                    "operator": "between",
                    "values": f"{sdmx_query.time_dimension_query.start_period},{sdmx_query.time_dimension_query.end_period}",
                }
            )

        return res

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
    _dimensions: t.Dict[str, SdmxDimension | VirtualDimension]
    _attributes: t.Dict[str, Sdmx21Attribute]
    _virtual_dimensions: t.Dict[str, VirtualDimension]
    _indicator_dimensions: t.Dict[str, SdmxCodeListDimension]
    _indicator_dimensions_required_for_query: list[str]
    _country_dimension: SdmxCodeListDimension | VirtualDimension | None
    _fixed_indicator: FixedItem | None
    # dimension_id -> {code_id -> code_name}
    _dim_values_id_2_name: dict[str, dict[str, str]] | None
    _attrib_values_id_2_name: dict[str, dict[str, str]] | None

    def __init__(
        self,
        entity_id: str,
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
        self._indicator_dimensions_required_for_query = []
        self._fixed_indicator = config.fixed_indicator
        self._virtual_dimensions = {}
        self._dim_values_id_2_name = None

        self._attributes = {attribute.entity_id: attribute for attribute in attributes}
        self._attrib_values_id_2_name = None

        if indicator_dimensions := config.indicator_dimensions:
            if self._fixed_indicator is not None:
                raise ValueError(
                    "fixed_indicator must not be provided if indicator_dimensions are present"
                )
            logger.info(f"dataset {self.entity_id}: using provided {indicator_dimensions=}")
            for dim_id in indicator_dimensions:
                dimension = self._dimensions[dim_id]
                if not isinstance(dimension, SdmxCodeListDimension):
                    raise TypeError(f"Indicator dimension must be code list dimension: {dimension}")
                self._indicator_dimensions[dim_id] = dimension
        else:
            if self._fixed_indicator is None:
                raise ValueError("either indicator_dimensions or fixed_indicator must be provided")
            logger.info(f"dataset {self.entity_id}: using fixed indicator: {self._fixed_indicator}")

        # indicator dimensions required for query
        diff = list(
            set(config.indicator_dimensions_required_for_query).difference(
                self._indicator_dimensions.keys()
            )
        )
        if diff:
            # check if we specified a fixed indicator
            if (
                len(diff) == 1
                and self._fixed_indicator is not None
                and diff[0] == self._fixed_indicator.id
            ):
                # we specified a fixed indicator
                self._indicator_dimensions_required_for_query = [diff[0]]
            else:
                raise ValueError(f"specified invalid indicators required for query: {diff} ")
        else:
            self._indicator_dimensions_required_for_query = list(
                config.indicator_dimensions_required_for_query
            )

        # virtual dimensions
        for virtual_dimension_config in config.virtual_dimensions:
            dimension = VirtualDimension(virtual_dimension_config)
            self._virtual_dimensions[dimension.entity_id] = dimension
            self._dimensions[dimension.entity_id] = dimension

        if country_dimension_id := config.country_dimension:
            dimension = self._dimensions[country_dimension_id]
            if not isinstance(dimension, SdmxCodeListDimension | VirtualDimension):
                raise TypeError(
                    f"Country dimension must be code list dimension or virtual dimension: {dimension}"
                )
            self._country_dimension = dimension
            self._country_dimension._alias = config.country_dimension_alias
        else:
            self._country_dimension = None

    async def updated_at(self, auth_context: AuthContext) -> datetime | None:
        if self.config.citation:
            if self.config.citation.last_updated:
                return parse(self.config.citation.last_updated)
        return None

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

    def _indicators_from_fixed_indicator(self) -> t.Sequence[ComplexIndicator]:
        if self._fixed_indicator is None:
            raise ValueError("fixed_indicator is None")

        # create a stub code to initialize CodeIndicator.
        # example of a correct URN:
        # urn:sdmx:org.sdmx.infomodel.codelist.Code=IMF_STA:CL_CPI_ANALYTICS_REPORTS_COUNTRY(1.0.0).512
        provider = self.entity_id.split(":")[0]
        urn = (
            f"urn:sdmx:org.sdmx.infomodel.codelist.Code={provider}:CL_STUB_INDICATOR(1.0.0)"
            f".{self._fixed_indicator.id}"
        )
        code = sdmx.model.common.Code(
            id=self._fixed_indicator.id,
            urn=urn,
            name=self._fixed_indicator.name,
            description=self._fixed_indicator.description,
        )
        code_category = CodeCategory(code=code, locale="en")
        return [ComplexIndicator([CodeIndicator(code_category)])]

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

    async def _load_indicator_combinations_to(
        self, file_path: str, auth_context: AuthContext
    ) -> None:
        indicator_ids = [x.entity_id for x in self.indicator_dimensions()]
        logger.info(
            f'{self.source_id}. Will extract available combinations for following indicator dimensions: {indicator_ids}'
        )

        time_start = time.time()

        query = DataSetAvailabilityQuery(dimensions_queries_dict={})
        avail_query_resp = await self.availability_query(query=query, auth_context=auth_context)

        avail_values = {
            dim: avail_query_resp.dimensions_queries_dict[dim].values for dim in indicator_ids
        }

        dim_2_avail_values_cnt_sorted = sorted(
            {k: len(v) for k, v in avail_values.items()}.items(), key=lambda x: x[1]
        )
        logger.info(f"{self.source_id} {dim_2_avail_values_cnt_sorted=}")
        order = [x[0] for x in dim_2_avail_values_cnt_sorted]

        series, queries_count = await self._get_available_series(
            cur_query={},
            cur_dim=order[0],
            cur_dim_avail_values=avail_values[order[0]],
            other_dims_to_fill=order[1:],
            queries_count=1,
            auth_context=auth_context,
        )
        logger.info(f'{self.source_id}. Extracted available indicator dimension combinations')
        logger.info(
            f"{self.source_id}. Number of series extracted: {len(series)}. Number of queries sent: {queries_count}"
        )

        elsapsed_time = time.time() - time_start
        logger.info(f'{self.source_id}. elapsed time: {elsapsed_time :.3f} sec')

        series_df = pd.DataFrame(series)
        series_df.sort_values(order, inplace=True)
        series_df.to_csv(file_path, index=False)
        logger.info(f"{self.source_id}. Saved indicator combinations to '{file_path}'")

        duplicates = series_df.duplicated().sum()
        if duplicates:
            raise ValueError(
                f"{self.source_id}. Found {duplicates} duplicates in the indicator combinations"
            )
        logger.info(f"{self.source_id}. No duplicates found in the indicator combinations")

    async def _get_or_load_indicator_combinations(self, auth_context: AuthContext) -> pd.DataFrame:
        file_name = escape_invalid_filename_chars(f"{self.source_id}.csv")

        if cache_dir := sdmx_settings.cache_dir:
            dir_name = os.path.join(cache_dir, sdmx_settings.indicator_combinations_subdir)
            file_path = os.path.join(str(dir_name), file_name)

            if not os.path.exists(file_path):
                os.makedirs(dir_name, exist_ok=True)
                # Create cache of available indicator combinations:
                logger.info(f"{self.source_id}. Indicator combinations cache not found.")
                await self._load_indicator_combinations_to(file_path, auth_context=auth_context)
            else:
                logger.info(f"{self.source_id}. Getting indicator combinations from cache.")

            return pd.read_csv(file_path, dtype=str)
        else:
            logger.info(
                f"{self.source_id}. Indicator combinations cache disabled. Loading to temp dir..."
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_path = os.path.join(tmp_dir, file_name)
                await self._load_indicator_combinations_to(file_path, auth_context)
                return pd.read_csv(file_path, dtype=str)

    async def _indicators_from_dimensions(
        self, auth_context: AuthContext
    ) -> t.Sequence[ComplexIndicator]:
        if not self._indicator_dimensions:
            raise ValueError("No indicator dimensions")

        df_avail_dim_combinations = await self._get_or_load_indicator_combinations(auth_context)

        # use the same columns order as in the list of indicator dimensions from dataset config.
        # NOTE: here we rely on the order of items in the dict, which is generally a bad practice.
        # however, it seems to work here, probably becase we don't modify the dict
        # after the pydantic model is created. and it seems to preser the insert order of keys.
        df_avail_dim_combinations = df_avail_dim_combinations[self._indicator_dimensions.keys()]

        dim_cat_id_2_model: dict[str, dict[str, CodeCategory]] = collections.defaultdict(dict)
        for dimension in self._indicator_dimensions.values():
            for code_category in dimension.available_values:
                dim_cat_id_2_model[dimension.entity_id][code_category.query_id] = code_category

        df_indicators = df_avail_dim_combinations.apply(  # type: ignore
            lambda row: ComplexIndicator(
                CodeIndicator(dim_cat_id_2_model[dim_id][row[dim_id]])
                for dim_id in df_avail_dim_combinations.columns
            ),
            axis=1,
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
    ) -> t.Tuple[SdmxQueryReadinessStatus, list[str]]:
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
        elif dimension.dimension_type == DimensionType.CATEGORY:
            self._append_category_query(query, dimension, result)
        elif dimension.is_time_dimension:
            self._append_time_dimension_query(query, result)

    def _to_sdmx_query(self, query: DataSetQuery) -> SdmxDataSetQuery:
        result = SdmxDataSetQuery.empty()
        # self._append_indicator_query_to_dataset_query(query, result)
        for dimension_query in query.dimensions_queries:
            if not dimension_query.values and dimension_query.operator != QueryOperator.ALL:
                continue
            dimension = self._dimensions[dimension_query.dimension_id]
            if isinstance(dimension, VirtualDimension):
                # virtual dimensions are not used in queries
                continue
            if dimension.dimension_type == DimensionType.CATEGORY:
                self._append_category_query(dimension_query, dimension, result)
        for dimension in self.non_virtual_dimensions():
            if dimension.is_mandatory and dimension.entity_id not in result:
                default_queries = self._config.dimension_default_queries.get(
                    dimension.entity_id, []
                )
                if default_queries:
                    for default_query in default_queries:
                        self._append_dimension_query(
                            DimensionQuery.from_query(default_query, dimension.entity_id), result
                        )
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
            if dimension.dimension_type == DimensionType.CATEGORY:
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

    def non_virtual_dimensions(self) -> t.Sequence[SdmxDimension]:
        return [dim for dim in self.dimensions() if not isinstance(dim, VirtualDimension)]

    def non_indicator_dimensions(self) -> list[SdmxDimension | VirtualDimension]:
        special_dimensions = set(sd.dimension_id for sd in self._config.special_dimensions)
        return [
            dimension
            for dimension in self.dimensions()
            if (dimension.entity_id not in self._indicator_dimensions)
            and (dimension.entity_id not in special_dimensions)
        ]

    def special_dimensions(self) -> dict[str, Dimension]:
        return {
            special_dimension.processor_id: self._dimensions[special_dimension.dimension_id]
            for special_dimension in self._config.special_dimensions
        }

    def indicator_dimensions(self) -> t.Sequence[SdmxCodeListDimension]:
        # TODO: does not support fixed indicator
        return list(self._indicator_dimensions.values())

    def indicator_dimensions_required_for_query(self) -> list[str]:
        return self._indicator_dimensions_required_for_query

    async def get_indicators(self, auth_context: AuthContext) -> t.Sequence[ComplexIndicator]:
        if self._fixed_indicator:
            return self._indicators_from_fixed_indicator()
        elif self._indicator_dimensions:
            return await self._indicators_from_dimensions(auth_context=auth_context)
        else:
            raise ValueError("No indicators")

    def country_dimension(self) -> CategoricalDimension | None:
        return self._country_dimension

    def get_dim_values_id_2_name_mapping(self) -> dict[str, dict[str, str]]:
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

    def get_attrib_values_id_2_name_mapping(self) -> dict[str, dict[str, str]]:
        if self._attrib_values_id_2_name is not None:
            return self._attrib_values_id_2_name

        self._attrib_values_id_2_name = {}
        for attrib in self.attributes():
            if not isinstance(attrib, Sdmx21CodeListAttribute):
                continue
            self._attrib_values_id_2_name[attrib.entity_id] = {
                code.query_id: code.name for code in attrib.code_list.codes()
            }

        return self._attrib_values_id_2_name

    def map_dim_values_id_2_name(
        self, value_ids: t.Iterable[str], dimension_name: str
    ) -> dict[str, str] | None:
        """Map dimension or attribute ids to their corresponding names."""
        id2name = (
            self.get_dim_values_id_2_name_mapping() | self.get_attrib_values_id_2_name_mapping()
        )

        cur_dim_id2name = id2name.get(dimension_name)
        if cur_dim_id2name is None:
            return None
        res = {_id: cur_dim_id2name.get(_id, '') for _id in value_ids}
        return res

    def map_dim_queries_2_names(self, queries: dict[str, list[str]]):
        """
        queries: {dimension_id: [list of value_ids]}
        """
        res = {}
        for dim_id, value_ids in queries.items():
            id2name = self.map_dim_values_id_2_name(value_ids, dim_id)
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

    def _availability_result_to_query(self, availability_result) -> DataSetAvailabilityQuery:
        constraints = list(availability_result.constraint.values())
        if len(constraints) != 1:
            raise ValueError("Unexpected quantity of constraints in structure message")
        constraint = constraints[0]
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

    def _get_query_url(self, response: requests.Response) -> str:
        url = response.url
        request_headers = response.request.headers
        if "Ocp-Apim-Subscription-Key" in request_headers:
            url += f"&subscription-key={request_headers['Ocp-Apim-Subscription-Key']}"
        return url

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

    def _data_msg_to_dataframe(self, data_msg: DataMessage) -> pd.DataFrame:
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

    def _include_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.include_attributes:
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

    async def _query_data(
        self, query: DataSetQuery, auth_context: AuthContext
    ) -> tuple[SdmxDataSetQuery, DataMessage]:
        status, missing_dimensions = self._evaluate_data_query_status(query)
        if status != SdmxQueryReadinessStatus.READY:
            raise ValueError(
                f"Query is not ready: {query}, missing dimensions: {missing_dimensions}"
            )
        sdmx_query = self._to_sdmx_query(query)
        logger.info(f"Querying dataset {self.entity_id} with {sdmx_query}")
        if sdmx_query.time_dimension_query is None:
            raise ValueError("Time dimension query is required")

        client = await self._datasource.create_sdmx_client(auth_context)
        data_msg: DataMessage = await client.data(
            agency_id=self._artefact.maintainer.id,  # type: ignore
            resource_id=self._artefact.id,
            version=self._artefact.version,  # type: ignore
            key=sdmx_query.get_key(),
            params=sdmx_query.get_params(),
            dsd=self._artefact.structure,
        )
        if not data_msg:
            raise ValueError("No data returned for the query")
        return sdmx_query, data_msg

    async def query(
        self, query: DataSetQuery, auth_context: AuthContext
    ) -> Sdmx21DataResponse | None:
        sdmx_query, data_msg = await self._query_data(query, auth_context)

        url = self._get_query_url(data_msg.response)  # type: ignore

        sdmx_pandas = self._data_msg_to_dataframe(data_msg)
        if sdmx_pandas.empty:
            req_url = data_msg.response.url if data_msg.response else None
            logger.warning(f"Empty response in dataset(id={self.entity_id}), url={req_url!r}")
            return None

        sdmx_pandas = self._include_attributes(sdmx_pandas)

        return Sdmx21DataResponse(
            dataset=self,
            df=sdmx_pandas,
            sdmx_query=sdmx_query,
            url=url,
        )

    def get_python_code(self, sdmx_query: SdmxDataSetQuery) -> str:
        if self._datasource.config.sdmx1_source:
            provider = self._datasource.config.sdmx1_source
        else:
            provider = self._artefact.maintainer.id  # type: ignore

        return self._get_python_query(
            provider=provider,
            resource_id=self.source_id,
            keys=sdmx_query.get_key(),
            params=sdmx_query.get_params(),
        )

    @staticmethod
    def _get_python_query(provider: str, resource_id: str, keys: dict, params: dict) -> str:
        return f'''\
# Uses the [sdmx1 library](https://pypi.org/project/sdmx1/)
# Install with:
# ```bash
# pip install sdmx1
# ```

import sdmx

provider = sdmx.Client("{provider}")
data_msg = provider.data(
    "{resource_id}",
    key={keys},
    params={params}
)\
'''
