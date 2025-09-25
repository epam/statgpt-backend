from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

from sdmx.message import StructureMessage
from sdmx.model.common import Codelist, Concept, ConceptScheme, MaintainableArtefact
from sdmx.model.v21 import Constraint, DataflowDefinition, DataStructureDefinition


class Urn(NamedTuple):
    agency_id: str
    resource_id: str
    version: str

    def short_urn(self) -> str:
        """Return a short URN representation."""
        return f"{self.agency_id}:{self.resource_id}({self.version})"

    def __repr__(self):
        return self.short_urn()

    @classmethod
    def for_artifact(cls, artifact: MaintainableArtefact) -> "Urn":
        if artifact.maintainer is None:
            raise ValueError(f"Artifact must have a maintainer to create a URN. {artifact}")

        return cls(
            agency_id=artifact.maintainer.id,
            resource_id=artifact.id,
            version=str(artifact.version),
        )


@dataclass
class ConceptIdentity:
    urn: Urn
    id: str

    def __repr__(self):
        return f"ConceptIdentity(urn={self.urn!r}, id={self.id!r})"

    @classmethod
    def from_sdmx1(cls, concept_identity: Concept) -> "ConceptIdentity":
        concept_schema = concept_identity.parent
        return cls(
            urn=Urn.for_artifact(concept_schema),  # type: ignore[arg-type]
            id=concept_identity.id,
        )


@dataclass  # for compatibility with sdmx1
class StructureMessage21:
    """
    Represents a SDMX2.1 StructureMessage.

    Custom class similar to the `sdmx1` StructureMessage, but with some fixes and improvements.
    """

    # Add more artifact collections as needed
    codelist: dict[Urn, Codelist] = field(default_factory=dict)
    concept_scheme: dict[Urn, ConceptScheme] = field(default_factory=dict)
    constraint: dict[Urn, Constraint] = field(default_factory=dict)
    dataflow: dict[Urn, DataflowDefinition] = field(default_factory=dict)
    structure: dict[Urn, DataStructureDefinition] = field(default_factory=dict)

    @classmethod
    def from_sdmx1(cls, structure_msg: StructureMessage) -> "StructureMessage21":
        """Convert a `sdmx1` StructureMessage to our custom StructureMessage21."""
        res = cls()

        mapping = [
            (res.codelist, structure_msg.codelist),
            (res.concept_scheme, structure_msg.concept_scheme),
            (res.constraint, structure_msg.constraint),
            (res.dataflow, structure_msg.dataflow),
            (res.structure, structure_msg.structure),
        ]

        for collection, items in mapping:
            for artifact in items.values():
                urn = Urn.for_artifact(artifact)
                collection[urn] = artifact  # type: ignore[index]

        return res

    def add_codelists(self, codelist: Iterable[Codelist]) -> None:
        for codelist in codelist:
            urn = Urn.for_artifact(codelist)
            self.codelist[urn] = codelist

    def add_concept_schemes(self, concept_scheme: Iterable[ConceptScheme]) -> None:
        for concept_scheme in concept_scheme:
            urn = Urn.for_artifact(concept_scheme)
            self.concept_scheme[urn] = concept_scheme

    def add_constraints(self, constraints: Iterable[Constraint]) -> None:
        for constraint in constraints:
            urn = Urn.for_artifact(constraint)
            self.constraint[urn] = constraint
