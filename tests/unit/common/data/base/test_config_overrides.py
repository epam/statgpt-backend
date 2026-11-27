import pytest
from pydantic import ValidationError

from statgpt.common.data.base.config import (
    IndicatorDimensionConfig,
    NonIndicatorDimensionConfig,
    SpecialDimensionConfig,
    TimePeriodDimensionConfig,
)
from statgpt.common.data.base.enums import SpecialNonIndicatorDimensions
from statgpt.common.data.sdmx.common.config import SdmxDataSetConfig


@pytest.mark.parametrize("by_alias", [False, True])
def test_typed_dimension_discriminators_survive_sparse_serialization(by_alias: bool) -> None:
    config = SdmxDataSetConfig.model_validate(
        {
            "urn": {"agencyId": "TEST", "resourceId": "DATA"},
            "dimensions": {
                "I": IndicatorDimensionConfig(is_required=False),
                "T": TimePeriodDimensionConfig(),
                "F": NonIndicatorDimensionConfig(subtype=SpecialNonIndicatorDimensions.FREQUENCY),
                "S": SpecialDimensionConfig(processor_id="test"),
            },
        }
    )
    dimension_type = "dimensionType" if by_alias else "dimension_type"
    expected = {
        "urn": {
            "agencyId" if by_alias else "agency_id": "TEST",
            "resourceId" if by_alias else "resource_id": "DATA",
        },
        "dimensions": {
            "I": {dimension_type: "INDICATOR", "isRequired" if by_alias else "is_required": False},
            "T": {dimension_type: "TIME_PERIOD"},
            "F": {dimension_type: "NON_INDICATOR", "subtype": "FREQUENCY"},
            "S": {dimension_type: "SPECIAL", "processorId" if by_alias else "processor_id": "test"},
        },
    }

    serialized = config.model_dump(mode="json", by_alias=by_alias, exclude_unset=True)

    assert serialized == expected
    restored = SdmxDataSetConfig.model_validate(serialized)
    assert restored.model_dump(mode="json", by_alias=by_alias, exclude_unset=True) == expected
    assert restored.indexing_hash == config.indexing_hash


def test_untyped_dimensions_still_require_discriminators() -> None:
    with pytest.raises(ValidationError) as exc:
        SdmxDataSetConfig.model_validate(
            {
                "urn": {"agencyId": "TEST", "resourceId": "DATA"},
                "dimensions": {
                    "I": {"isRequired": False},
                    "T": {},
                    "F": {"subtype": "FREQUENCY"},
                    "S": {"processorId": "test"},
                },
            }
        )

    assert len(exc.value.errors()) == 4
    assert all(error["type"] == "union_tag_not_found" for error in exc.value.errors())
