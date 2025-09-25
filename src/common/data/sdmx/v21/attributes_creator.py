from sdmx.model import common
from sdmx.model.v21 import DataStructureDefinition

from common.config.logging import logger
from common.data.sdmx.common.codelist import BaseSdmxCodeList, InMemoryCodeList
from common.data.sdmx.v21.attribute import (
    Sdmx21Attribute,
    Sdmx21CodeListAttribute,
    Sdmx21StringAttribute,
)
from common.data.sdmx.v21.schemas import StructureMessage21, Urn


class Sdmx21AttributesCreator:

    def __init__(self, structure_message: StructureMessage21, urn: Urn, locale: str):
        self._structure_message = structure_message
        self._urn = urn
        self._locale = locale

    async def create_attributes(self) -> list[Sdmx21Attribute]:
        return self._create_attributes()

    @property
    def _dsd(self) -> DataStructureDefinition:
        return self._structure_message.dataflow[self._urn].structure

    @property
    def _sdmx_attributes(self) -> list[common.DataAttribute]:
        return self._dsd.attributes.components

    def _create_attributes(self) -> list[Sdmx21Attribute]:
        dims = [
            self._create_attribute_from_concept(attribute=attribute)
            for attribute in self._sdmx_attributes
        ]
        return dims

    def _get_concept_for(self, attribute: common.DataAttribute) -> common.Concept:
        concept_identity = attribute.concept_identity
        if not concept_identity:
            raise ValueError(f"{attribute=} does not contain required concept_identity")
        if not concept_identity.parent:
            raise ValueError(f"{attribute=} does not contain required concept_identity.parent")

        urn = Urn.for_artifact(concept_identity.parent)  # type: ignore[arg-type]
        schema = self._structure_message.concept_scheme[urn]
        return schema.items[concept_identity.id]

    def _create_attribute_from_concept(self, attribute: common.DataAttribute) -> Sdmx21Attribute:
        concept = self._get_concept_for(attribute)
        representation = concept.core_representation
        if not representation:
            representation = attribute.local_representation
            if not representation:
                raise ValueError(f"Concept {concept} has neither core nor local representation")

        name = concept.name[self._locale]
        description = concept.description.localizations.get(self._locale)

        if representation.enumerated is not None:
            # This log is produced on every DbSdmx21DataSourceHandler.get_dataset() method call.
            # It is expected during dataset indexing -
            # so that we see which representations are used for each attribute.
            # But get_dataset() is called often when answering the user query (by chat facade).
            # TODO: either control when to produce this log depending on the context, or remove it.
            # logger.info(f"Creating code list attribute from core_representation for {attribute=}")
            result_attribute = self._create_code_list_attribute(
                code_list_ref=representation.enumerated,
                attribute=attribute,
                name=name,
                description=description,
            )
        elif representation.non_enumerated:
            facets = representation.non_enumerated
            facet = facets[0]  # for now, only the first facet is processed
            logger.debug(f"Attribute facet_type={facet.value_type} for {attribute=}")

            result_attribute = Sdmx21StringAttribute(
                attribute=attribute,
                name=name,
                description=description,
            )
        else:
            raise ValueError(f"Unsupported {representation=} for {attribute=}")

        return result_attribute

    def _create_sdmx_code_list_attribute(
        self,
        attribute: common.DataAttribute,
        name: str,
        description: str | None,
        code_list: BaseSdmxCodeList,
    ) -> Sdmx21CodeListAttribute:
        return Sdmx21CodeListAttribute(attribute, name, description, code_list)

    def _create_code_list_attribute(
        self,
        code_list_ref: common.ItemScheme | None,
        attribute: common.DataAttribute,
        name: str,
        description: str | None,
    ):
        if not isinstance(code_list_ref, common.Codelist):
            raise ValueError(
                f"Could only create code list attribute from Codelist type, got: {type(code_list_ref)}"
            )

        # Actually, `code_list_ref` is a valid code list, but it does not contain the data,
        # since we are loading the codelist separately.
        code_list = self._structure_message.codelist[Urn.for_artifact(code_list_ref)]

        indexed_codelist = InMemoryCodeList(code_list, self._locale)
        return self._create_sdmx_code_list_attribute(attribute, name, description, indexed_codelist)
