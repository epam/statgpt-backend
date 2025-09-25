from sdmx.message import StructureMessage
from sdmx.model.v21 import DataStructureDefinition

from common.settings.dataflow_loader import DataflowLoaderSettings
from common.utils import async_utils

from .schemas import ConceptIdentity, StructureMessage21, Urn
from .sdmx_client import AsyncSdmxClient


class DataflowLoader:

    _SETTINGS = DataflowLoaderSettings()

    def __init__(self, client: AsyncSdmxClient):
        self._client: AsyncSdmxClient = client

    async def load_structure_message(self, urn: Urn) -> StructureMessage21:
        dataflow_msg = await self._load_dataflow(urn)
        result_message = StructureMessage21.from_sdmx1(dataflow_msg)

        schemes = await self._load_concept_schemes(result_message.dataflow[urn].structure)
        for scheme_msg in schemes:
            result_message.add_concept_schemes(scheme_msg.concept_scheme.values())

        code_lists = await self._load_code_lists(result_message, urn)
        for code_list_msg in code_lists:
            result_message.add_codelists(code_list_msg.codelist.values())

        constraint_message = await self._load_constraints(urn)
        result_message.add_constraints(constraint_message.constraint.values())

        return result_message

    async def _load_dataflow(self, urn: Urn) -> StructureMessage:
        return await self._client.dataflow(
            agency_id=urn.agency_id,
            resource_id=urn.resource_id,
            version=urn.version,
            params={'references': 'datastructure'},
            use_cache=True,
        )

    async def _load_constraints(self, urn: Urn) -> StructureMessage:
        return await self._client.availableconstraint(
            agency_id=urn.agency_id,
            resource_id=urn.resource_id,
            version=urn.version,
            use_cache=True,
        )

    async def _load_code_lists(
        self, dataflow_msg: StructureMessage21, urn: Urn
    ) -> list[StructureMessage]:
        code_lists = self._get_code_lists(dataflow_msg, urn)

        tasks = [
            self._client.codelist(
                agency_id=code_list.agency_id,
                resource_id=code_list.resource_id,
                version=code_list.version,
                use_cache=True,
            )
            for code_list in code_lists
        ]
        return await async_utils.gather_with_concurrency(
            self._SETTINGS.code_list_concurrency_limit, *tasks
        )

    def _get_code_lists(self, dataflow_msg: StructureMessage21, urn: Urn) -> set[Urn]:
        dsd: DataStructureDefinition = dataflow_msg.dataflow[urn].structure

        code_lists = set()
        for concept in self._get_concepts_from(dsd):
            concept_scheme = dataflow_msg.concept_scheme[concept.urn]
            concept_item = concept_scheme.items[concept.id]

            core_repr = concept_item.core_representation
            if core_repr is not None and core_repr.enumerated is not None:
                code_lists.add(Urn.for_artifact(core_repr.enumerated))

        return code_lists

    async def _load_concept_schemes(self, dsd: DataStructureDefinition) -> list[StructureMessage]:
        schemas = set(concept.urn for concept in self._get_concepts_from(dsd))

        tasks = [
            self._client.conceptscheme(
                agency_id=concept_scheme.agency_id,
                resource_id=concept_scheme.resource_id,
                version=concept_scheme.version,
                use_cache=True,
            )
            for concept_scheme in schemas
        ]
        return await async_utils.gather_with_concurrency(
            self._SETTINGS.concept_scheme_concurrency_limit, *tasks
        )

    @staticmethod
    def _get_concepts_from(dsd: DataStructureDefinition) -> list[ConceptIdentity]:
        """Extract concept schemes from the data structure definition."""
        concepts = []

        for attr in dsd.attributes.components:
            if attr.concept_identity is not None:
                concepts.append(ConceptIdentity.from_sdmx1(attr.concept_identity))

        for dim in dsd.dimensions.components:
            if dim.concept_identity is not None:
                concepts.append(ConceptIdentity.from_sdmx1(dim.concept_identity))

        return concepts
