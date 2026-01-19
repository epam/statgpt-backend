"""Unit tests for the IndexingField marker and hash computation utilities."""

from typing import Annotated

import pytest
from pydantic import BaseModel, ConfigDict, Field, alias_generators

from statgpt.common.data.base.indexing import (
    IndexingField,
    IndexingHashMixin,
    _collect_indexing_fields,
    _has_indexing_marker,
    compute_indexing_hash,
)


class TestHasIndexingMarker:
    """Tests for the _has_indexing_marker function."""

    def test_field_with_marker_returns_true(self) -> None:
        class Model(BaseModel):
            marked: Annotated[str, IndexingField()] = ""

        field_info = Model.model_fields["marked"]
        assert _has_indexing_marker(field_info) is True

    def test_field_without_marker_returns_false(self) -> None:
        class Model(BaseModel):
            unmarked: str = ""

        field_info = Model.model_fields["unmarked"]
        assert _has_indexing_marker(field_info) is False

    def test_field_with_other_annotations_returns_false(self) -> None:
        class Model(BaseModel):
            other: Annotated[str, "some_other_marker"] = ""

        field_info = Model.model_fields["other"]
        assert _has_indexing_marker(field_info) is False

    def test_field_with_multiple_annotations_including_marker(self) -> None:
        class Model(BaseModel):
            multi: Annotated[str, "other", IndexingField(), "another"] = ""

        field_info = Model.model_fields["multi"]
        assert _has_indexing_marker(field_info) is True


class TestCollectIndexingFields:
    """Tests for the _collect_indexing_fields function."""

    def test_collects_marked_fields_only(self) -> None:
        class Model(BaseModel):
            marked: Annotated[str, IndexingField()] = "value1"
            unmarked: str = "value2"

        model = Model()
        fields = dict(_collect_indexing_fields(model))

        assert fields == {"marked": "value1"}

    def test_handles_none_values(self) -> None:
        class Model(BaseModel):
            nullable: Annotated[str | None, IndexingField()] = None

        model = Model()
        fields = dict(_collect_indexing_fields(model))

        assert fields == {"nullable": None}

    def test_handles_nested_model_with_markers(self) -> None:
        class Inner(BaseModel):
            inner_marked: Annotated[str, IndexingField()] = "inner_value"
            inner_unmarked: str = "ignored"

        class Outer(BaseModel):
            nested: Annotated[Inner, IndexingField()] = Field(default_factory=Inner)

        model = Outer()
        fields = dict(_collect_indexing_fields(model))

        assert fields == {"nested.inner_marked": "inner_value"}

    def test_handles_nested_model_without_markers(self) -> None:
        """When nested model has no markers, nothing is yielded from it.

        This is the expected behavior - fields must be explicitly marked.
        Note: dict[str, Model] handling includes full dump for unmarked models
        to support the dimensions pattern where the dict itself is marked.
        """

        class Inner(BaseModel):
            field1: str = "a"
            field2: str = "b"

        class Outer(BaseModel):
            nested: Annotated[Inner, IndexingField()] = Field(default_factory=Inner)

        model = Outer()
        fields = dict(_collect_indexing_fields(model))

        assert fields == {}

    def test_handles_dict_of_models(self) -> None:
        class Item(BaseModel):
            name: Annotated[str, IndexingField()] = ""
            ignored: str = ""

        class Container(BaseModel):
            items: Annotated[dict[str, Item], IndexingField()] = Field(default_factory=dict)

        model = Container(items={"a": Item(name="first"), "b": Item(name="second")})
        fields = dict(_collect_indexing_fields(model))

        assert fields == {"items": {"a": {"name": "first"}, "b": {"name": "second"}}}

    def test_dict_keys_are_sorted(self) -> None:
        class Item(BaseModel):
            value: Annotated[int, IndexingField()] = 0

        class Container(BaseModel):
            items: Annotated[dict[str, Item], IndexingField()] = Field(default_factory=dict)

        model = Container(items={"z": Item(value=1), "a": Item(value=2), "m": Item(value=3)})
        fields = dict(_collect_indexing_fields(model))

        # Keys should be sorted for determinism
        assert list(fields["items"].keys()) == ["a", "m", "z"]

    def test_handles_dict_of_primitives(self) -> None:
        class Model(BaseModel):
            mapping: Annotated[dict[str, int], IndexingField()] = Field(default_factory=dict)

        model = Model(mapping={"x": 1, "y": 2})
        fields = dict(_collect_indexing_fields(model))

        assert fields == {"mapping": {"x": 1, "y": 2}}

    def test_handles_list_of_models(self) -> None:
        class Item(BaseModel):
            name: Annotated[str, IndexingField()] = ""

        class Container(BaseModel):
            items: Annotated[list[Item], IndexingField()] = Field(default_factory=list)

        model = Container(items=[Item(name="first"), Item(name="second")])
        fields = dict(_collect_indexing_fields(model))

        assert fields == {"items": [{"name": "first"}, {"name": "second"}]}

    def test_handles_list_of_primitives(self) -> None:
        class Model(BaseModel):
            values: Annotated[list[int], IndexingField()] = Field(default_factory=list)

        model = Model(values=[1, 2, 3])
        fields = dict(_collect_indexing_fields(model))

        assert fields == {"values": [1, 2, 3]}

    def test_handles_primitive_types(self) -> None:
        class Model(BaseModel):
            string_field: Annotated[str, IndexingField()] = "text"
            int_field: Annotated[int, IndexingField()] = 42
            bool_field: Annotated[bool, IndexingField()] = True
            float_field: Annotated[float, IndexingField()] = 3.14

        model = Model()
        fields = dict(_collect_indexing_fields(model))

        assert fields == {
            "string_field": "text",
            "int_field": 42,
            "bool_field": True,
            "float_field": 3.14,
        }


class TestComputeIndexingHash:
    """Tests for the compute_indexing_hash function."""

    def test_returns_string_hash(self) -> None:
        class Model(BaseModel):
            field: Annotated[str, IndexingField()] = "value"

        model = Model()
        result = compute_indexing_hash(model)

        assert isinstance(result, str)
        assert result.isdigit() or result.lstrip("-").isdigit()  # CRC32 can be negative

    def test_same_model_produces_same_hash(self) -> None:
        class Model(BaseModel):
            field: Annotated[str, IndexingField()] = "value"

        model1 = Model()
        model2 = Model()

        assert compute_indexing_hash(model1) == compute_indexing_hash(model2)

    def test_different_values_produce_different_hashes(self) -> None:
        class Model(BaseModel):
            field: Annotated[str, IndexingField()] = ""

        model1 = Model(field="value1")
        model2 = Model(field="value2")

        assert compute_indexing_hash(model1) != compute_indexing_hash(model2)

    def test_unmarked_fields_do_not_affect_hash(self) -> None:
        class Model(BaseModel):
            marked: Annotated[str, IndexingField()] = "same"
            unmarked: str = ""

        model1 = Model(unmarked="different1")
        model2 = Model(unmarked="different2")

        assert compute_indexing_hash(model1) == compute_indexing_hash(model2)

    def test_hash_is_deterministic_with_dict_field(self) -> None:
        """Dict ordering should not affect hash due to sort_keys=True."""

        class Model(BaseModel):
            data: Annotated[dict[str, int], IndexingField()] = Field(default_factory=dict)

        # Create dicts with different insertion orders
        model1 = Model(data={"a": 1, "b": 2, "c": 3})
        model2 = Model(data={"c": 3, "a": 1, "b": 2})

        assert compute_indexing_hash(model1) == compute_indexing_hash(model2)

    def test_empty_model_produces_consistent_hash(self) -> None:
        class Model(BaseModel):
            pass

        model = Model()
        hash1 = compute_indexing_hash(model)
        hash2 = compute_indexing_hash(model)

        assert hash1 == hash2


class TestIndexingHashMixin:
    """Tests for the IndexingHashMixin class."""

    def test_provides_indexing_hash_property(self) -> None:
        class Model(BaseModel, IndexingHashMixin):
            field: Annotated[str, IndexingField()] = "value"

        model = Model()
        assert hasattr(model, "indexing_hash")
        assert isinstance(model.indexing_hash, str)

    def test_indexing_hash_matches_compute_function(self) -> None:
        class Model(BaseModel, IndexingHashMixin):
            field: Annotated[str, IndexingField()] = "value"

        model = Model()
        assert model.indexing_hash == compute_indexing_hash(model)

    def test_mixin_requires_base_model(self) -> None:
        class NotAModel(IndexingHashMixin):
            pass

        obj = NotAModel()
        with pytest.raises(TypeError, match="must be used with Pydantic BaseModel"):
            _ = obj.indexing_hash


class TestInheritance:
    """Tests for inheritance behavior with IndexingField markers."""

    def test_child_inherits_parent_markers(self) -> None:
        class Parent(BaseModel, IndexingHashMixin):
            parent_field: Annotated[str, IndexingField()] = "parent"

        class Child(Parent):
            child_field: Annotated[str, IndexingField()] = "child"

        model = Child()
        fields = dict(_collect_indexing_fields(model))

        assert fields == {"parent_field": "parent", "child_field": "child"}

    def test_child_can_add_markers_to_inherited_unmarked_fields(self) -> None:
        class Parent(BaseModel, IndexingHashMixin):
            unmarked_in_parent: str = "value"

        class Child(Parent):
            unmarked_in_parent: Annotated[str, IndexingField()] = "value"

        parent_fields = dict(_collect_indexing_fields(Parent()))
        child_fields = dict(_collect_indexing_fields(Child()))

        assert parent_fields == {}
        assert child_fields == {"unmarked_in_parent": "value"}

    def test_different_subclasses_can_have_different_hashes(self) -> None:
        class Parent(BaseModel, IndexingHashMixin):
            common: Annotated[str, IndexingField()] = "same"

        class Child1(Parent):
            unique: Annotated[str, IndexingField()] = "child1"

        class Child2(Parent):
            unique: Annotated[str, IndexingField()] = "child2"

        model1 = Child1()
        model2 = Child2()

        assert model1.indexing_hash != model2.indexing_hash


class TestRealWorldScenarios:
    """Tests simulating real-world usage patterns from the codebase."""

    def test_dimension_config_like_structure(self) -> None:
        """Simulates the DataSetConfig.dimensions structure."""

        class DimensionConfig(BaseModel):
            dimension_type: Annotated[str, IndexingField()] = "INDICATOR"
            alias: Annotated[str | None, IndexingField()] = None
            is_required: bool = False  # Not marked - should not affect hash

        class DataSetConfig(BaseModel, IndexingHashMixin):
            dimensions: Annotated[dict[str, DimensionConfig], IndexingField()] = Field(
                default_factory=dict
            )

        config1 = DataSetConfig(
            dimensions={
                "IND1": DimensionConfig(dimension_type="INDICATOR", alias="indicator1"),
                "DIM1": DimensionConfig(dimension_type="NON_INDICATOR", alias=None),
            }
        )

        # Same config with different is_required (unmarked) should have same hash
        config2 = DataSetConfig(
            dimensions={
                "IND1": DimensionConfig(
                    dimension_type="INDICATOR", alias="indicator1", is_required=True
                ),
                "DIM1": DimensionConfig(
                    dimension_type="NON_INDICATOR", alias=None, is_required=True
                ),
            }
        )

        assert config1.indexing_hash == config2.indexing_hash

    def test_nested_config_hierarchy(self) -> None:
        """Simulates nested config structures like IndexerConfig -> IndexerIndicatorConfig."""

        class AnnotationConfig(BaseModel):
            description: Annotated[str, IndexingField()] = ""

        class IndicatorConfig(BaseModel):
            unpack: Annotated[bool, IndexingField()] = False
            annotations: Annotated[AnnotationConfig | None, IndexingField()] = None

        class IndexerConfig(BaseModel, IndexingHashMixin):
            description: Annotated[str, IndexingField()] = ""
            indicator: Annotated[IndicatorConfig, IndexingField()] = Field(
                default_factory=IndicatorConfig
            )

        config1 = IndexerConfig(
            description="Test",
            indicator=IndicatorConfig(
                unpack=True, annotations=AnnotationConfig(description="annotation")
            ),
        )

        config2 = IndexerConfig(
            description="Test",
            indicator=IndicatorConfig(
                unpack=True, annotations=AnnotationConfig(description="different")
            ),
        )

        # Different annotation description should produce different hash
        assert config1.indexing_hash != config2.indexing_hash

    def test_urn_reference_like_structure(self) -> None:
        """Simulates UrnReference from SDMX config."""

        class UrnReference(BaseModel):
            agency_id: Annotated[str, IndexingField()] = ""
            resource_id: Annotated[str, IndexingField()] = ""
            version: Annotated[str, IndexingField()] = "latest"

        class DataSetConfig(BaseModel, IndexingHashMixin):
            urn: Annotated[UrnReference, IndexingField()] = Field(default_factory=UrnReference)
            include_attributes: list[str] | None = None  # Not marked

        config1 = DataSetConfig(
            urn=UrnReference(agency_id="AGENCY", resource_id="FLOW", version="1.0"),
            include_attributes=["attr1", "attr2"],
        )

        config2 = DataSetConfig(
            urn=UrnReference(agency_id="AGENCY", resource_id="FLOW", version="1.0"),
            include_attributes=None,  # Different but unmarked
        )

        assert config1.indexing_hash == config2.indexing_hash

        config3 = DataSetConfig(
            urn=UrnReference(agency_id="AGENCY", resource_id="FLOW", version="2.0"),
        )

        assert config1.indexing_hash != config3.indexing_hash


class TestWithNonIndexingFieldsFrom:
    """Tests for the with_non_indexing_fields_from method."""

    def test_merges_marked_from_resolved_and_unmarked_from_current(self) -> None:
        """IndexingField-marked fields come from resolved, others from current."""

        class Config(BaseModel, IndexingHashMixin):
            marked: Annotated[str, IndexingField()] = ""
            unmarked: str = ""

        current = Config(marked="current_value", unmarked="current_unmarked")
        resolved = Config(marked="resolved_value", unmarked="resolved_unmarked")

        result = resolved.with_non_indexing_fields_from(current)

        assert result.model_dump() == {
            "marked": "resolved_value",
            "unmarked": "current_unmarked",
        }

    def test_camel_case_aliases_with_model_dump(self) -> None:
        """Should handle camelCase aliases correctly when using model_dump."""

        class Config(BaseModel, IndexingHashMixin):
            model_config = ConfigDict(
                alias_generator=alias_generators.to_camel, populate_by_name=True
            )
            is_official: bool = False
            indexing_field: Annotated[str, IndexingField()] = ""

        current = Config(is_official=True, indexing_field="current")
        resolved = Config(is_official=False, indexing_field="resolved")

        result = resolved.with_non_indexing_fields_from(current)

        assert result.model_dump(mode="json", by_alias=True) == {
            "isOfficial": True,  # From current (not marked)
            "indexingField": "resolved",  # From resolved (marked)
        }

    def test_nested_model_recursive_merge(self) -> None:
        """Nested models should be recursively merged."""

        class Inner(BaseModel):
            marked_inner: Annotated[str, IndexingField()] = ""
            unmarked_inner: str = ""

        class Outer(BaseModel, IndexingHashMixin):
            nested: Annotated[Inner, IndexingField()] = Field(default_factory=Inner)
            outer_unmarked: str = ""

        current = Outer(
            nested=Inner(marked_inner="current_marked", unmarked_inner="current_unmarked"),
            outer_unmarked="current_outer",
        )
        resolved = Outer(
            nested=Inner(marked_inner="resolved_marked", unmarked_inner="resolved_unmarked"),
            outer_unmarked="resolved_outer",
        )

        result = resolved.with_non_indexing_fields_from(current)

        assert result.model_dump() == {
            "nested": {
                "marked_inner": "resolved_marked",  # From resolved (marked)
                "unmarked_inner": "current_unmarked",  # From current (not marked)
            },
            "outer_unmarked": "current_outer",  # From current (not marked)
        }

    def test_dict_of_models_per_key_merge(self) -> None:
        """dict[str, BaseModel] fields should merge per-key."""

        class DimensionConfig(BaseModel):
            dimension_type: Annotated[str, IndexingField()] = "INDICATOR"
            is_required: bool = False  # Not marked

        class DataSetConfig(BaseModel, IndexingHashMixin):
            dimensions: Annotated[dict[str, DimensionConfig], IndexingField()] = Field(
                default_factory=dict
            )
            citation: str | None = None  # Not marked

        current = DataSetConfig(
            dimensions={
                "DIM1": DimensionConfig(dimension_type="INDICATOR", is_required=True),
                "DIM2": DimensionConfig(dimension_type="NON_INDICATOR", is_required=True),
            },
            citation="Current citation",
        )
        resolved = DataSetConfig(
            dimensions={
                "DIM1": DimensionConfig(dimension_type="INDICATOR", is_required=False),
                "DIM2": DimensionConfig(dimension_type="NON_INDICATOR", is_required=False),
            },
            citation="Resolved citation",
        )

        result = resolved.with_non_indexing_fields_from(current)

        assert result.model_dump() == {
            "dimensions": {
                "DIM1": {"dimension_type": "INDICATOR", "is_required": True},
                "DIM2": {"dimension_type": "NON_INDICATOR", "is_required": True},
            },
            "citation": "Current citation",
        }

    def test_preserves_resolved_dimension_keys(self) -> None:
        """Should preserve dimension keys from resolved, not current."""

        class DimensionConfig(BaseModel):
            dimension_type: Annotated[str, IndexingField()] = "INDICATOR"

        class DataSetConfig(BaseModel, IndexingHashMixin):
            dimensions: Annotated[dict[str, DimensionConfig], IndexingField()] = Field(
                default_factory=dict
            )

        current = DataSetConfig(
            dimensions={
                "DIM1": DimensionConfig(dimension_type="INDICATOR"),
                "NEW_DIM": DimensionConfig(dimension_type="NON_INDICATOR"),  # New key
            },
        )
        resolved = DataSetConfig(
            dimensions={
                "DIM1": DimensionConfig(dimension_type="INDICATOR"),
                "DIM2": DimensionConfig(dimension_type="TIME_PERIOD"),  # Not in current
            },
        )

        result = resolved.with_non_indexing_fields_from(current)

        # Should have keys from resolved only (dimensions is marked)
        assert result.model_dump() == {
            "dimensions": {
                "DIM1": {"dimension_type": "INDICATOR"},
                "DIM2": {"dimension_type": "TIME_PERIOD"},
            },
        }

    def test_handles_none_values(self) -> None:
        """Should handle None values correctly."""

        class Config(BaseModel, IndexingHashMixin):
            nullable_marked: Annotated[str | None, IndexingField()] = None
            nullable_unmarked: str | None = None

        current = Config(nullable_marked="current", nullable_unmarked="current")
        resolved = Config(nullable_marked=None, nullable_unmarked=None)

        result = resolved.with_non_indexing_fields_from(current)

        assert result.model_dump() == {
            "nullable_marked": None,  # From resolved (marked)
            "nullable_unmarked": "current",  # From current (unmarked)
        }

    def test_realistic_sdmx_like_config(self) -> None:
        """Test with a realistic SDMX-like configuration structure."""

        class UrnReference(BaseModel):
            agency_id: Annotated[str, IndexingField()] = ""
            resource_id: Annotated[str, IndexingField()] = ""
            version: Annotated[str, IndexingField()] = ""

        class DimensionConfig(BaseModel):
            model_config = ConfigDict(
                alias_generator=alias_generators.to_camel, populate_by_name=True
            )
            dimension_type: Annotated[str, IndexingField()] = "NON_INDICATOR"
            alias: Annotated[str | None, IndexingField()] = None
            is_required: bool = False  # Not marked
            default_queries: list[dict] | None = None  # Not marked

        class DataSetConfig(BaseModel, IndexingHashMixin):
            model_config = ConfigDict(
                alias_generator=alias_generators.to_camel, populate_by_name=True
            )
            urn: Annotated[UrnReference, IndexingField()] = Field(default_factory=UrnReference)
            dimensions: Annotated[dict[str, DimensionConfig], IndexingField()] = Field(
                default_factory=dict
            )
            is_official: bool = False  # Not marked
            citation: str | None = None  # Not marked
            pinned_columns: list[str] = Field(default_factory=list)  # Not marked

        current = DataSetConfig(
            urn=UrnReference(agency_id="CURRENT", resource_id="FLOW", version="2.0"),
            dimensions={
                "INDICATOR": DimensionConfig(
                    dimension_type="INDICATOR",
                    alias="ind",
                    is_required=True,
                    default_queries=[{"value": "GDP"}],
                ),
                "COUNTRY": DimensionConfig(
                    dimension_type="NON_INDICATOR",
                    alias=None,
                    is_required=True,
                    default_queries=None,
                ),
            },
            is_official=True,
            citation="Updated citation 2024",
            pinned_columns=["INDICATOR", "COUNTRY"],
        )
        resolved = DataSetConfig(
            urn=UrnReference(agency_id="RESOLVED", resource_id="FLOW", version="1.0"),
            dimensions={
                "INDICATOR": DimensionConfig(
                    dimension_type="INDICATOR",
                    alias="indicator",
                    is_required=False,
                    default_queries=None,
                ),
                "COUNTRY": DimensionConfig(
                    dimension_type="NON_INDICATOR",
                    alias="country",
                    is_required=False,
                    default_queries=None,
                ),
            },
            is_official=False,
            citation="Original citation",
            pinned_columns=[],
        )

        result = resolved.with_non_indexing_fields_from(current)

        assert result.model_dump(mode="json", by_alias=True) == {
            # URN from resolved (marked)
            "urn": {
                "agency_id": "RESOLVED",
                "resource_id": "FLOW",
                "version": "1.0",
            },
            # Dimensions with merged fields
            "dimensions": {
                "INDICATOR": {
                    "dimensionType": "INDICATOR",  # From resolved (marked)
                    "alias": "indicator",  # From resolved (marked)
                    "isRequired": True,  # From current (not marked)
                    "defaultQueries": [{"value": "GDP"}],  # From current (not marked)
                },
                "COUNTRY": {
                    "dimensionType": "NON_INDICATOR",  # From resolved (marked)
                    "alias": "country",  # From resolved (marked)
                    "isRequired": True,  # From current (not marked)
                    "defaultQueries": None,  # From current (not marked)
                },
            },
            # Non-indexing top-level fields from current
            "isOfficial": True,
            "citation": "Updated citation 2024",
            "pinnedColumns": ["INDICATOR", "COUNTRY"],
        }


class TestIndexingHashMixinWithNonIndexingFieldsFromTypeError:
    """Tests for the with_non_indexing_fields_from method type error."""

    def test_mixin_requires_base_model_for_merge(self) -> None:
        class NotAModel(IndexingHashMixin):
            pass

        obj = NotAModel()
        with pytest.raises(TypeError, match="must be used with Pydantic BaseModel"):
            obj.with_non_indexing_fields_from(obj)  # type: ignore[arg-type]
