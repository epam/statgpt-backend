"""Tests for SDMX 2.1 attribute creation: representation resolution and its string default."""

from dataclasses import dataclass, field

from sdmx.model import common
from sdmx.model.v21 import DataflowDefinition, DataStructureDefinition

from statgpt.common.data.base import AttributeType
from statgpt.common.data.sdmx.v21.attribute import (
    Sdmx21Attribute,
    Sdmx21CodeListAttribute,
    Sdmx21StringAttribute,
)
from statgpt.common.data.sdmx.v21.attributes_creator import Sdmx21AttributesCreator
from statgpt.common.data.sdmx.v21.schemas import StructureMessage21, Urn

AGENCY = "TEST_AGENCY"
LOCALE = "en"

CODELIST_ID = "CL_TEST_OBS_STATUS"
CORE_VERSION = "1.0"
LOCAL_VERSION = "1.5"

# Two versions of the same codelist that genuinely disagree: the newer one, referenced by the
# attribute's local representation, carries a code the older one (referenced by the concept's core
# representation) never had.
CODE_IN_BOTH = "CODE_A"
CODE_IN_LOCAL_ONLY = "CODE_B"


@dataclass
class AttributeSpec:
    """One attribute of the data structure, with the representations declared for it."""

    id: str
    name: str
    local: common.Representation | None = None
    core: common.Representation | None = None
    codelists: list[common.Codelist] = field(default_factory=list)


def _codelist(version: str, code_ids: list[str]) -> common.Codelist:
    codelist = common.Codelist(id=CODELIST_ID, maintainer=common.Agency(id=AGENCY), version=version)
    for code_id in code_ids:
        codelist.append(common.Code(id=code_id, name=f"{code_id} ({version})"))
    return codelist


def _string_representation() -> common.Representation:
    return common.Representation(
        non_enumerated=[common.Facet(value_type=common.FacetValueType.string)]
    )


def _coded_attribute_spec() -> AttributeSpec:
    """An attribute whose concept declares an enumerated core representation, and nothing local."""
    codelist = _codelist(CORE_VERSION, [CODE_IN_BOTH])
    return AttributeSpec(
        id="OBS_STATUS",
        name="Observation status",
        core=common.Representation(enumerated=codelist),
        codelists=[codelist],
    )


def _build_message(specs: list[AttributeSpec]) -> tuple[StructureMessage21, Urn]:
    agency = common.Agency(id=AGENCY)

    concept_scheme = common.ConceptScheme(id="CS_TEST", maintainer=agency, version="1.0")
    dsd = DataStructureDefinition(id="DSD_TEST", maintainer=agency, version="1.0")

    for spec in specs:
        concept = common.Concept(id=spec.id, name=spec.name, parent=concept_scheme)
        concept.core_representation = spec.core
        concept_scheme.append(concept)
        dsd.attributes.append(
            common.DataAttribute(
                id=spec.id, concept_identity=concept, local_representation=spec.local
            )
        )

    dataflow = DataflowDefinition(
        id="DSD_TEST@DF_TEST", maintainer=agency, version="1.0", structure=dsd
    )

    message = StructureMessage21()
    urn = Urn.for_artifact(dataflow)
    message.dataflow[urn] = dataflow
    message.structure[Urn.for_artifact(dsd)] = dsd
    message.concept_scheme[Urn.for_artifact(concept_scheme)] = concept_scheme
    message.add_codelists([codelist for spec in specs for codelist in spec.codelists])
    return message, urn


def _create_attributes(specs: list[AttributeSpec]) -> dict[str, Sdmx21Attribute]:
    message, urn = _build_message(specs)
    attributes = Sdmx21AttributesCreator(message, urn, LOCALE)._create_attributes()
    return {attribute.entity_id: attribute for attribute in attributes}


def test_attribute_without_representation_defaults_to_string() -> None:
    """An attribute with no representation on either side resolves to a string attribute.

    Regression test: raising here lost the whole dataflow, because every attribute is built in
    one comprehension. SDMX 2.1 defines the default representation as untyped `xs:string`, so
    `VAR` in OECD.SDD.TPS:DSD_SDBSBD_ISIC4@DF_BD_API is a valid string-valued attribute.
    """
    spec = AttributeSpec(id="VAR", name="Variable", local=None, core=None)

    attributes = _create_attributes([spec])

    attribute = attributes["VAR"]
    assert isinstance(attribute, Sdmx21StringAttribute)
    assert attribute.attribute_type == AttributeType.STRING
    assert attribute.name == "Variable"


def test_local_string_facet_resolves_to_string_attribute() -> None:
    """An uncoded local representation still resolves to a string attribute.

    This is the `LOCAL_AREA_NAME` shape, where the provider declares
    `TextFormat textType="String"` explicitly.
    """
    spec = AttributeSpec(
        id="LOCAL_AREA_NAME", name="Local area name", local=_string_representation()
    )

    attributes = _create_attributes([spec])

    assert isinstance(attributes["LOCAL_AREA_NAME"], Sdmx21StringAttribute)


def test_core_representation_used_when_no_local_representation() -> None:
    """The concept's core representation remains the fallback when the attribute declares none."""
    attributes = _create_attributes([_coded_attribute_spec()])

    attribute = attributes["OBS_STATUS"]
    assert isinstance(attribute, Sdmx21CodeListAttribute)
    assert attribute.attribute_type == AttributeType.CATEGORY
    assert attribute.code_list.code_list.version == CORE_VERSION


def test_local_representation_overrides_core_representation() -> None:
    """An attribute's local representation wins over the concept's core representation.

    SDMX 2.1 inherits from the concept only when the component declares no local representation.
    Preferring the core representation resolved the attribute to the older codelist version, so
    any code added in a later version was missing from it.
    """
    local_codelist = _codelist(LOCAL_VERSION, [CODE_IN_BOTH, CODE_IN_LOCAL_ONLY])
    core_codelist = _codelist(CORE_VERSION, [CODE_IN_BOTH])
    spec = AttributeSpec(
        id="OBS_STATUS",
        name="Observation status",
        local=common.Representation(enumerated=local_codelist),
        core=common.Representation(enumerated=core_codelist),
        codelists=[local_codelist, core_codelist],
    )

    attributes = _create_attributes([spec])

    attribute = attributes["OBS_STATUS"]
    assert isinstance(attribute, Sdmx21CodeListAttribute)
    assert attribute.code_list.code_list.version == LOCAL_VERSION
    assert CODE_IN_LOCAL_ONLY in attribute.code_list


def test_representation_less_attribute_does_not_abort_the_other_attributes() -> None:
    """A dataflow shaped like DSD_SDBSBD_ISIC4 keeps all four of its attributes."""
    coded_ids = ["DECIMALS", "OBS_STATUS", "UNIT_MULT"]
    specs = []
    for attribute_id in coded_ids:
        codelist = _codelist(CORE_VERSION, [CODE_IN_BOTH])
        specs.append(
            AttributeSpec(
                id=attribute_id,
                name=attribute_id.capitalize(),
                local=common.Representation(enumerated=codelist),
                core=common.Representation(enumerated=codelist),
                codelists=[codelist],
            )
        )
    specs.append(AttributeSpec(id="VAR", name="Variable"))

    attributes = _create_attributes(specs)

    assert sorted(attributes) == sorted(coded_ids + ["VAR"])
    assert all(
        isinstance(attributes[attribute_id], Sdmx21CodeListAttribute) for attribute_id in coded_ids
    )
    assert isinstance(attributes["VAR"], Sdmx21StringAttribute)
