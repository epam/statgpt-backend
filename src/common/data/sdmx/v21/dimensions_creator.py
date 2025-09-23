from sdmx.model import common
from sdmx.model.v21 import ContentConstraint, DataStructureDefinition

from common.config.logging import logger
from common.data.sdmx.common import (
    InMemoryCodeList,
    SdmxCodeListDimension,
    SdmxDimension,
    SdmxTimeDimension,
)
from common.data.sdmx.common.codelist import BaseSdmxCodeList
from common.data.sdmx.v21.schemas import StructureMessage21, Urn


class DimensionsCreator:

    def __init__(self, structure_message: StructureMessage21, urn: Urn, locale: str):
        self._structure_message = structure_message
        self._urn = urn
        self._locale = locale

    async def create_dimensions(self) -> list[SdmxDimension]:
        return self._create_dimensions()

    @property
    def _dsd(self) -> DataStructureDefinition:
        return self._structure_message.dataflow[self._urn].structure

    @property
    def _sdmx_dimensions(self) -> list[common.DimensionComponent]:
        return self._dsd.dimensions.components

    def _find_time_period_dimension(self):
        """Apply number of heuristics to find TimePeriod dimension"""
        # TODO: TimePeriod dimension is defined in the yaml config, it must be used here!

        def _check_dimension_dtype(dimension: common.DimensionComponent):
            return isinstance(dimension, common.TimeDimension)

        def _check_dimension_id(dimension: common.DimensionComponent):
            return dimension.id == "TIME_PERIOD"

        def _check_facet_value_type(dimension: common.DimensionComponent):
            ci = dimension.concept_identity
            if ci is None:
                return False
            cr = ci.core_representation
            if cr is None:
                return False
            facet = cr.non_enumerated[0]
            return facet.value_type == common.FacetValueType.observationalTimePeriod

        predicates = [_check_dimension_dtype, _check_dimension_id, _check_facet_value_type]
        for predicate in predicates:
            for dim in self._sdmx_dimensions:
                if predicate(dim):
                    return dim
        return None

    def _create_dimensions(self) -> list[SdmxDimension]:

        time_period_dim = self._find_time_period_dimension()
        if time_period_dim is None:
            raise ValueError(
                f"Couldn't find TimePeriod dimension. Got the following dimension components: {self._sdmx_dimensions!r}"
            )

        dims = [
            self._create_dimension_from_concept(
                dimension=dimension,
                time_dimension=(dimension.id == time_period_dim.id),
            )
            for dimension in self._sdmx_dimensions
        ]
        return dims

    def _get_concept_for(self, dimension: common.DimensionComponent) -> common.Concept:
        concept_identity = dimension.concept_identity
        if not concept_identity:
            raise ValueError(f"{dimension=} does not contain required concept_identity")
        if not concept_identity.parent:
            raise ValueError(f"{dimension=} does not contain required concept_identity.parent")

        urn = Urn.for_artifact(concept_identity.parent)  # type: ignore[arg-type]
        schema = self._structure_message.concept_scheme[urn]
        return schema.items[concept_identity.id]

    def _create_dimension_from_concept(
        self, dimension: common.DimensionComponent, time_dimension: bool
    ) -> SdmxDimension:
        concept = self._get_concept_for(dimension)
        representation = concept.core_representation
        if not representation:
            representation = dimension.local_representation
            if not representation:
                raise ValueError(f"Concept {concept} has neither core nor local representation")

        name = concept.name[self._locale]
        description = concept.description.localizations.get(self._locale)

        if time_dimension:
            if not isinstance(dimension, common.TimeDimension):
                raise ValueError(f"dimension must be of TimeDimension type. got: {type(dimension)}")

            return self._create_time_period_dimension(
                dimension=dimension,
                name=name,
                description=description,
                representation=representation,
            )

        if not isinstance(dimension, common.Dimension):
            raise ValueError(f"dimension must be of Dimension type. got: {type(dimension)}")

        if representation.enumerated is not None:
            # This log is produced on every DbSdmx21DataSourceHandler.get_dataset() method call.
            # It is expected during dataset indexing -
            # so that we see which representations are used for each dimension.
            # But get_dataset() is called often when answering the user query (by chat facade).
            # TODO: either control when to produce this log depending on the context, or remove it.
            # logger.info(f"Creating code list dimension from core_representation for {dimension=}")
            result_dimension = self._create_code_list_dimension(
                code_list_ref=representation.enumerated,
                dimension=dimension,
                name=name,
                description=description,
            )
        elif representation.non_enumerated:
            facets = representation.non_enumerated
            facet = facets[0]  # for now, only the first facet is processed

            if facet.value_type == common.FacetValueType.string:
                # TODO: same note as for log above
                # logger.warning(
                #     f"Creating code list dimension from local_representation for {dimension=}"
                # )
                if local_representation := dimension.local_representation:
                    result_dimension = self._create_code_list_dimension(
                        code_list_ref=local_representation.enumerated,
                        dimension=dimension,
                        name=name,
                        description=description,
                    )
                else:
                    raise ValueError(f"Dimension {dimension} has no local representation")
            else:
                raise ValueError(
                    f"Failed to build SdmxDimension for {dimension=}. {facet.value_type=}"
                )
        else:
            raise ValueError(f"Unsupported {representation=} for {dimension=}")

        return result_dimension

    def _create_sdmx_code_list_dimension(
        self,
        dimension: common.DimensionComponent,
        name: str,
        description: str | None,
        code_list: BaseSdmxCodeList,
    ) -> SdmxCodeListDimension:
        available_codes = self._get_available_dimension_values(dimension)
        return SdmxCodeListDimension(dimension, name, description, code_list, available_codes)

    def _create_code_list_dimension(
        self,
        code_list_ref: common.ItemScheme | None,
        dimension: common.Dimension,
        name: str,
        description: str | None,
    ):
        if not isinstance(code_list_ref, common.Codelist):
            raise ValueError(
                f"Could only create code list dimension from Codelist type, got: {type(code_list_ref)}"
            )

        # Actually, `code_list_ref` is a valid code list, but it does not contain the data,
        # since we are loading the codelist separately.
        code_list = self._structure_message.codelist[Urn.for_artifact(code_list_ref)]

        indexed_codelist = InMemoryCodeList(code_list, self._locale)
        return self._create_sdmx_code_list_dimension(dimension, name, description, indexed_codelist)

    @staticmethod
    def _create_time_period_dimension(
        dimension: common.TimeDimension,
        name: str,
        description: str | None,
        representation: common.Representation,
    ):
        facets = representation.non_enumerated
        facet = facets[0]  # for now, only the first facet is processed

        allowed_facet_dtypes = {
            common.FacetValueType.observationalTimePeriod,
            common.FacetValueType.string,
        }
        if facet.value_type not in allowed_facet_dtypes:
            raise ValueError(
                f"{facet.value_type=} is unavailable. must be one of {allowed_facet_dtypes=}"
            )

        logger.debug(f"Creating SdmxTimeDimension for {dimension=}")
        result_dimension = SdmxTimeDimension(dimension, name, description, time_dimension=True)
        return result_dimension

    def _get_available_dimension_values(self, dimension: common.DimensionComponent) -> set[str]:
        constraints = list(self._structure_message.constraint.values())
        if len(constraints) != 1:
            raise ValueError("Unexpected quantity of constraints in structure message")
        constraint: ContentConstraint = constraints[0]  # type: ignore[assignment]

        if len(constraint.data_content_region) != 1:
            raise ValueError("Unexpected quantity of cube-regions in constraint")

        cube_region = constraint.data_content_region[0]

        member_dict = cube_region.member
        if dimension.id not in member_dict.keys():
            raise ValueError(f"Missing dimension({dimension.id}) value in data content constraint")
        return set([item.value for item in member_dict[dimension.id].values])  # type: ignore
