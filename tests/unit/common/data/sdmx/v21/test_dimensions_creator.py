"""Tests for SDMX 2.1 dimension creation, in particular representation precedence."""

from sdmx.model import common
from sdmx.model.v21 import (
    ContentConstraint,
    CubeRegion,
    DataflowDefinition,
    DataStructureDefinition,
    MemberSelection,
    MemberValue,
)

from statgpt.common.data.sdmx.common import SdmxCodeListDimension
from statgpt.common.data.sdmx.v21.dimensions_creator import DimensionsCreator
from statgpt.common.data.sdmx.v21.schemas import StructureMessage21, Urn

AGENCY = "TEST_AGENCY"
LOCALE = "en"

CODELIST_ID = "CL_TEST_MEASURE"
LOCAL_VERSION = "1.5"
CORE_VERSION = "1.0"

# Two versions of the same codelist that genuinely disagree: the newer one, referenced by the
# dimension's local representation, carries a code the older one (referenced by the concept's core
# representation) never had.
CODE_IN_BOTH = "CODE_A"
CODE_IN_LOCAL_ONLY = "CODE_B"


def _codelist(version: str, code_ids: list[str]) -> common.Codelist:
    codelist = common.Codelist(id=CODELIST_ID, maintainer=common.Agency(id=AGENCY), version=version)
    for code_id in code_ids:
        codelist.append(common.Code(id=code_id, name=f"{code_id} ({version})"))
    return codelist


def _build_message(
    local_codelist: common.Codelist | None, core_codelist: common.Codelist
) -> tuple[StructureMessage21, Urn]:
    agency = common.Agency(id=AGENCY)

    concept_scheme = common.ConceptScheme(id="CS_TEST", maintainer=agency, version="1.0")
    measure_concept = common.Concept(id="MEASURE", name="Measure", parent=concept_scheme)
    measure_concept.core_representation = common.Representation(enumerated=core_codelist)
    time_concept = common.Concept(id="TIME_PERIOD", name="Time period", parent=concept_scheme)
    time_concept.core_representation = common.Representation(
        non_enumerated=[common.Facet(value_type=common.FacetValueType.observationalTimePeriod)]
    )
    concept_scheme.append(measure_concept)
    concept_scheme.append(time_concept)

    measure_dim = common.Dimension(
        id="MEASURE",
        order=1,
        concept_identity=measure_concept,
        local_representation=(
            common.Representation(enumerated=local_codelist) if local_codelist else None
        ),
    )
    time_dim = common.TimeDimension(
        id="TIME_PERIOD",
        order=2,
        concept_identity=time_concept,
        local_representation=common.Representation(
            non_enumerated=[common.Facet(value_type=common.FacetValueType.observationalTimePeriod)]
        ),
    )

    dsd = DataStructureDefinition(id="DSD_TEST", maintainer=agency, version="1.0")
    dsd.dimensions.append(measure_dim)
    dsd.dimensions.append(time_dim)

    dataflow = DataflowDefinition(
        id="DSD_TEST@DF_TEST", maintainer=agency, version="1.0", structure=dsd
    )

    # Availability lists both codes - it is version-agnostic, it just enumerates code ids.
    constraint = ContentConstraint(
        id="CR_A_DSD_TEST@DF_TEST",
        maintainer=agency,
        version="1.0",
        role=None,
        data_content_region=[
            CubeRegion(
                included=True,
                member={
                    measure_dim: MemberSelection(
                        values_for=measure_dim,
                        values=[
                            MemberValue(value=CODE_IN_BOTH),
                            MemberValue(value=CODE_IN_LOCAL_ONLY),
                        ],
                    )
                },
            )
        ],
    )

    message = StructureMessage21()
    urn = Urn.for_artifact(dataflow)
    message.dataflow[urn] = dataflow
    message.structure[Urn.for_artifact(dsd)] = dsd
    message.concept_scheme[Urn.for_artifact(concept_scheme)] = concept_scheme
    message.constraint[Urn.for_artifact(constraint)] = constraint
    message.add_codelists([core_codelist] + ([local_codelist] if local_codelist else []))
    return message, urn


def _create_measure_dimension(
    local_codelist: common.Codelist | None, core_codelist: common.Codelist
) -> SdmxCodeListDimension:
    message, urn = _build_message(local_codelist, core_codelist)
    creator = DimensionsCreator(message, urn, LOCALE, aliases={})
    dimensions = creator._create_dimensions()

    measure = next(dim for dim in dimensions if dim.entity_id == "MEASURE")
    assert isinstance(measure, SdmxCodeListDimension)
    return measure


def test_local_representation_overrides_core_representation() -> None:
    """A dimension's local representation wins over the concept's core representation.

    Regression test: preferring the core representation resolved the dimension to the older
    codelist version, so any code added in a later version raised a bare KeyError during
    indexing even though availability advertised it.
    """
    local_codelist = _codelist(LOCAL_VERSION, [CODE_IN_BOTH, CODE_IN_LOCAL_ONLY])
    core_codelist = _codelist(CORE_VERSION, [CODE_IN_BOTH])

    measure = _create_measure_dimension(local_codelist, core_codelist)

    assert measure.code_list.code_list.version == LOCAL_VERSION
    assert CODE_IN_LOCAL_ONLY in measure.code_list
    # The code the availability constraint advertises must resolve rather than raise.
    assert measure.code_list[CODE_IN_LOCAL_ONLY].query_id == CODE_IN_LOCAL_ONLY
    assert {value.query_id for value in measure.available_values} == {
        CODE_IN_BOTH,
        CODE_IN_LOCAL_ONLY,
    }


def test_core_representation_used_when_no_local_representation() -> None:
    """The core representation remains the fallback when the dimension declares none."""
    core_codelist = _codelist(CORE_VERSION, [CODE_IN_BOTH])

    measure = _create_measure_dimension(None, core_codelist)

    assert measure.code_list.code_list.version == CORE_VERSION
    assert CODE_IN_BOTH in measure.code_list
