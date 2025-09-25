import math
import typing as t
import warnings

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel

# This conditional import is used to avoid circular imports
if t.TYPE_CHECKING:
    from common.data.sdmx import Sdmx21DataSet


def df_2_plotly_grid(df: pd.DataFrame, round_digits: int | None = None) -> go.Figure:
    if round_digits is not None:
        # copy of df with limited number of significant digits
        df = df.round(round_digits)
    num_rows = min(len(df), 7)  # limit number of rows to 7
    row_height = 28 * 2
    header_height = 28
    total_height = (
        num_rows * row_height
    ) * 3 // 2 + header_height  # heuristics for long names in the table
    # Create Plotly Table
    figure = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=df.columns,
                    fill_color="#333942",
                    font=dict(family="var(--font-inter)", color="#7F8792"),
                    align="left",
                    height=header_height,
                    line=dict(color="#222932"),
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color="#141A23",
                    align='left',
                    height=row_height,
                    line=dict(color="#222932"),
                ),
            )
        ],
        layout=go.Layout(
            template="plotly_dark",
            showlegend=False,
            width=1000,
            height=total_height,
            font=dict(family="var(--font-inter)", size=14, color="#F3F4F6"),
            margin=dict(t=0, b=0, r=0, l=0),
            annotations=[dict(text="", showarrow=False, xref="paper", yref="paper")],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return figure


class PlotsConfig(BaseModel):
    col_height: int = 400
    col_width: int = 600
    max_num_cols: int = 1
    horizontal_spacing: float = 0.1
    vertical_spacing: float = 0.15
    title_font_size: int = 14
    subplot_title_font_size: int = 12


class PlotlyGraphBuilder:
    _df: pd.DataFrame
    _config: PlotsConfig
    _dimensions_ids_2_names: dict[str, dict[str, str]]
    _id_2_name_cols: dict[str, str]
    _value_col: str

    _NAME_COL_SUFFIX: str = "_Name"

    def __init__(
        self,
        df: pd.DataFrame,
        dataset: 'Sdmx21DataSet',
        *,
        config: PlotsConfig | None = None,
        value_col: str = "value",
        min_time_period_values: int = 3,
    ) -> None:
        self._dataset = dataset
        self._config = config or PlotsConfig()
        self._value_col = value_col
        df = self._validate_df_then_copy(df)
        self._id_2_name_cols = self._map_id_2_name_cols(df)
        self._dimensions_ids_2_names = self._map_dimensions_ids_2_names(df)
        self._df = self._set_multiindex(df)
        self._min_time_period_values = min_time_period_values

    @property
    def _indicator_id_col(self) -> str:
        try:
            return self._dataset.indicator_dimensions()[0].entity_id
        except IndexError:
            raise ValueError("There is no at least one indicator dimension")

    @property
    def _time_period_col(self) -> str:
        return self._dataset.get_time_dimension().entity_id

    @property
    def _country_col(self) -> str | None:
        dimension = self._dataset.country_dimension()
        return dimension.entity_id if dimension else None

    @property
    def _indicator_name_col(self) -> str:
        return self._indicator_id_col + self._NAME_COL_SUFFIX

    def plot_for_all_indicators(
        self,
    ) -> list[go.Figure]:
        plots = []
        for indicator, df_view in self._df_views_per_indicator():
            plot = self._plot_for_indicator(indicator, df_view)
            if plot:
                plots.append(plot)
        return plots

    @staticmethod
    def _calc_num_rows_and_cols_in_grid(count: int, max_num_cols: int) -> tuple[int, int]:
        if count < 0:
            raise ValueError(f"`count` can't be negative: {count}")
        if count == 0:
            return 0, 0
        num_cols = min(max_num_cols, math.ceil(math.sqrt(count)))
        num_rows = math.ceil(count / num_cols)
        return num_rows, num_cols

    def _df_views_per_indicator(self) -> t.Iterator[tuple[str, pd.DataFrame]]:
        unique_indicators = self._df[self._indicator_id_col].unique()
        for indicator in unique_indicators:
            df_view = self._df[self._df[self._indicator_id_col] == indicator]
            df_view = df_view.drop(columns=[self._indicator_id_col])[self._value_col]
            # "TIME_PERIOD" must be the last index level
            index_levels_order = [c for c in df_view.index.names if c != self._time_period_col] + [
                self._time_period_col
            ]
            df_view.index = df_view.index.reorder_levels(index_levels_order)
            yield indicator, df_view

    def _generate_titles(self, indicator: str, multiindex: pd.Index) -> tuple[str, list[str]]:
        unambiguous_dimensions: dict[str, str] = {}

        dimension: str
        for dimension in multiindex.names:  # type: ignore[assignment]
            dimension_values = multiindex.get_level_values(dimension).unique()
            if len(dimension_values) != 1:
                continue
            unambiguous_dimensions[dimension] = str(dimension_values[0])

        indicator_verbose_name = self._dimensions_ids_2_names[self._indicator_id_col][indicator]
        title = f"[{indicator}] {indicator_verbose_name}"
        dimensions_lines = "<br>".join(
            [
                f"{dim}={self._dimensions_ids_2_names[dim][dim_value]}"
                for dim, dim_value in unambiguous_dimensions.items()
            ]
        )
        if dimensions_lines:
            title += f"<br>{dimensions_lines}"

        subplots_titles = []
        for multiid in multiindex:
            ambiguous_names: list[str] = [
                str(name) for name in multiindex.names if str(name) not in unambiguous_dimensions
            ]
            ambiguous_multiid = [
                value
                for name, value in zip(multiindex.names, multiid)
                if name not in unambiguous_dimensions
            ]
            subplots_titles.append(
                self._subplot_title_from_multiid(ambiguous_multiid, ambiguous_names)
            )
        return title, subplots_titles

    def _subplot_title_from_multiid(self, multiid: list[t.Any], multiindex_names: list[str]) -> str:
        return "<br>".join(
            f"{dim_id_col}={self._dimensions_ids_2_names[dim_id_col][dim_value]}"
            for dim_id_col, dim_value in zip(multiindex_names, multiid)
        )

    def _map_dimensions_ids_2_names(self, df: pd.DataFrame) -> dict[str, dict[str, str]]:
        dimensions_ids_2_names = dict()
        for id_col, name_col in self._id_2_name_cols.items():
            dimensions_ids_2_names[id_col] = (
                df[[id_col, name_col]].drop_duplicates().set_index(id_col)[name_col].to_dict()
            )
        return dimensions_ids_2_names

    def _map_id_2_name_cols(self, df: pd.DataFrame) -> dict[str, str]:
        id_2_name_cols: dict[str, str | None] = {
            c: None
            for c in df.columns
            if not c.endswith(self._NAME_COL_SUFFIX)
            and c != self._time_period_col
            and c != self._value_col
        }

        for c in df.columns:
            if not c.endswith(self._NAME_COL_SUFFIX) or c == self._value_col:
                continue
            expected_id_col = c[: -len(self._NAME_COL_SUFFIX)]
            if expected_id_col not in id_2_name_cols:
                warnings.warn(
                    f"Matching id column {expected_id_col!r} for name column {c!r} is not found in the data frame, so they will be ignored."
                )
                continue
            id_2_name_cols[expected_id_col] = c

        for id_col, name_col in id_2_name_cols.items():
            if name_col is not None:
                continue
            expected_name_col = f"{id_col}{self._NAME_COL_SUFFIX}"
            warnings.warn(
                f"Matching name column {expected_name_col!r} for id column {id_col!r} is not found in the data frame, so they will be ignored."
            )

        cleaned_id_2_name_cols: dict[str, str] = {
            k: v for k, v in id_2_name_cols.items() if v is not None
        }
        return cleaned_id_2_name_cols

    def _add_scatter_per_each_country(
        self, figure: go.Figure, df: pd.DataFrame, row: int, col: int
    ):
        for country in df[self._country_col].unique():
            observations_df = df[df[self._country_col] == country].drop(columns=self._country_col)
            self._add_scatter(
                figure,
                observations_df,
                name=self._dimensions_ids_2_names[self._country_col][country],  # type: ignore
                row=row,
                col=col,
            )

    def _add_scatter(self, figure: go.Figure, df, name: str, row: int, col: int):
        figure.add_trace(
            go.Scatter(x=df[self._time_period_col], y=df[self._value_col], name=name),
            row=row,
            col=col,
        )

    def _filter_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        filter out series which doesn't have enough observations to create scatter
        """
        group_sizes = df.groupby(level=df.index.names).size()
        mask = group_sizes > self._min_time_period_values
        df = df[df.index.isin(group_sizes[mask].index)]
        return df

    def _plot_for_indicator(
        self,
        indicator: str,
        df_view: pd.DataFrame,
    ) -> t.Optional[go.Figure]:
        df_view = df_view.reset_index(self._time_period_col)

        filtered_df = self._filter_series(df_view)
        if filtered_df.empty:
            return None

        if self._country_col:
            filtered_df = filtered_df.reset_index(self._country_col)

        multiindex = filtered_df.index.unique()

        num_plots = len(multiindex)
        num_rows, num_cols = self._calc_num_rows_and_cols_in_grid(
            num_plots, max_num_cols=self._config.max_num_cols
        )
        title, subplot_titles = self._generate_titles(indicator, multiindex)
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=self._config.horizontal_spacing,
            vertical_spacing=self._config.horizontal_spacing / num_rows,
        )

        for counter, (multiid, subplot_title) in enumerate(zip(multiindex, subplot_titles)):
            df = filtered_df.loc[multiid].sort_index()
            self._cast_time_col_to_datetime(df)

            row = (counter // num_cols) + 1
            col = (counter % num_cols) + 1

            if self._country_col:
                self._add_scatter_per_each_country(fig, df, row, col)
            else:
                self._add_scatter(fig, df, subplot_title, row, col)

        n_subplot_title_lines = subplot_titles[0].count("<br>") + 1
        n_title_lines = title.count("<br>") + 1
        subplot_title_size = 2 * self._config.subplot_title_font_size * n_subplot_title_lines
        subplot_height = self._config.col_height + 1.5 * (
            subplot_title_size + 2 * self._config.subplot_title_font_size
        )
        title_size = 2 * self._config.title_font_size * n_title_lines
        fig.update_annotations(font_size=self._config.subplot_title_font_size)

        fig.update_layout(
            title={
                "text": title,
                "font": {
                    "size": self._config.title_font_size,
                },
                "y": 0.97,
                "xanchor": 'left',
                "yanchor": 'top',
            },
            margin={
                "t": title_size + subplot_title_size,
            },
            width=num_cols * self._config.col_width,
            height=num_rows * subplot_height,
            showlegend=False,
        )
        return fig

    def _cast_time_col_to_datetime(self, df: pd.DataFrame):
        with warnings.catch_warnings():
            # TODO: here we ignore the warning,
            # however it's better to ensure that date format is parsed correctly.
            # 1.probably we can know the format in advance
            # (seems unlikely, because we may have data in different frequencies)
            # 2. or we can be verbose about date parsing failures.
            # 3. alternatively can use following code:
            #
            # # Convert index to datetime and coerce errors
            # seq.index = pd.to_datetime(seq.index, errors='coerce')
            # # Check for any NaT (Not a Time) values
            # if seq.index.isna().any():
            #     print("There were some parsing errors.")
            #     print(seq[seq.index.isna()])
            #
            warnings.simplefilter("ignore")
            df[self._time_period_col] = pd.to_datetime(df[self._time_period_col])

    def _set_multiindex(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=list(self._id_2_name_cols.values()), inplace=True)
        non_value_cols = [c for c in df.columns if c != self._value_col]
        df.set_index(non_value_cols, inplace=True)
        df.reset_index([self._indicator_id_col], inplace=True)
        return df

    def _validate_df_then_copy(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._indicator_id_col not in df.columns:
            raise ValueError(
                f"Required indicator id column {self._indicator_id_col!r} not found in the data frame."
            )
        if self._indicator_name_col not in df.columns:
            raise ValueError(
                f"Required indicator name column {self._indicator_name_col!r} not found in the data frame."
            )
        return df.copy()
