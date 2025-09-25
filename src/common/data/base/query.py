import typing as t

from pydantic import BaseModel, Field, model_validator

from .enums import QueryOperator


class Query(BaseModel):
    values: list[str] = Field(
        description="The values to filter on, for categorical dimensions it's the category IDs",
        min_length=0,
    )
    operator: QueryOperator = Field(description="The operator to use for filtering")

    @property
    def is_all_selected(self) -> bool:
        return self.operator == QueryOperator.ALL

    def is_empty(self):
        return not (bool(self.values) or self.is_all_selected)


class DimensionQuery(Query):
    dimension_id: str = Field(description="The ID of the dimension")
    is_default: bool = Field(default=False, description="Whether the query is the default one")

    @model_validator(mode='after')
    def _check_flags(self):
        if self.is_default and self.is_all_selected:
            raise ValueError(
                "Only one of is_default and is_all_selected can be True."
                f" Got: is_default={self.is_default}, is_all_selected={self.is_all_selected}"
            )
        return self

    @classmethod
    def from_query(cls, query: Query, dimension_id: str) -> "DimensionQuery":
        return cls(values=query.values, operator=query.operator, dimension_id=dimension_id)

    def to_query(self) -> Query:
        return Query(values=self.values, operator=self.operator)

    @classmethod
    def from_default_query(cls, query: Query, dimension_id: str) -> "DimensionQuery":
        return cls(
            values=query.values, operator=query.operator, dimension_id=dimension_id, is_default=True
        )


class DataSetQuery(BaseModel):
    dimensions_queries: t.List[DimensionQuery] = Field(description="The queries for dimensions")
    is_valid: bool = Field(True, description="Whether the query is valid")

    @property
    def dimensions_queries_dict(self) -> t.Dict[str, Query]:
        return {
            query.dimension_id: Query(values=query.values, operator=query.operator)
            for query in self.dimensions_queries
        }


class DataSetAvailabilityQuery(BaseModel):
    # NOTE: this class is more generic than availability queries only - rename?
    dimensions_queries_dict: t.Dict[str, Query] = Field(
        description="The queries for dimensions", default_factory=dict
    )

    @classmethod
    def from_dimension_queries_list(
        cls, queries: list[DimensionQuery]
    ) -> "DataSetAvailabilityQuery":
        dims = [q.dimension_id for q in queries]
        if len(dims) != len(set(dims)):
            raise ValueError(f"Duplicate dimension IDs in queries list: {dims}")
        return cls(
            dimensions_queries_dict={
                dim_query.dimension_id: dim_query.to_query() for dim_query in queries
            }
        )

    @property
    def dimensions_queries(self) -> list[DimensionQuery]:
        return [
            DimensionQuery.from_query(query, dimension_id)
            for dimension_id, query in self.dimensions_queries_dict.items()
        ]

    def add_dimension_query(self, query: DimensionQuery) -> None:
        """Add or overwrite existing query for specified dimension."""
        self.dimensions_queries_dict[query.dimension_id] = Query(
            values=query.values, operator=query.operator
        )

    def is_empty(self) -> bool:
        """True if there are no non-empty dimension queries"""
        for query in self.dimensions_queries_dict.values():
            if not query.is_empty():
                return False
        return True

    def __contains__(self, item: str) -> bool:
        return item in self.dimensions_queries_dict

    def __getitem__(self, item: str) -> DimensionQuery:
        if item in self.dimensions_queries_dict:
            return DimensionQuery.from_query(self.dimensions_queries_dict[item], item)
        raise KeyError(item)

    def filter(self, other: "DataSetAvailabilityQuery") -> "DataSetAvailabilityQuery":
        """
        User 'other' query to filter current one.
        """

        result = self.__class__()
        for dimension_id, query in self.dimensions_queries_dict.items():
            if query.operator != QueryOperator.IN:
                result.add_dimension_query(DimensionQuery.from_query(query, dimension_id))
                continue
            if dimension_id not in other.dimensions_queries_dict:
                continue
            other_query = other.dimensions_queries_dict[dimension_id]
            common_values = list(set(query.values).intersection(other_query.values))
            result.add_dimension_query(
                DimensionQuery(
                    values=common_values,
                    operator=QueryOperator.IN,
                    dimension_id=dimension_id,
                )
            )
        return result
