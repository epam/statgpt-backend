"""Resolving the effective SDMX 2.1 representation of a component.

SDMX 2.1 resolves a component's representation in three steps: the component's own
`LocalRepresentation`, then the `CoreRepresentation` of the concept it identifies, then the
standard's default of untyped `xs:string`. The last step is normative - `ConceptType` in
`SDMXStructureConcept.xsd` states that a concept without a `TextFormat` or core representation
"is assumed to be represented by any set of valid characters (corresponding to the xs:string
datatype of W3C XML Schema)" - and `AttributeType` explicitly allows a component to declare
neither representation.

`sdmx1` models the first two steps as optional fields and never supplies the third, so a caller
that treats a missing representation as an error rejects structures the standard allows. One such
component used to abort the whole dataflow.
"""

import logging

from sdmx.model import common

_log = logging.getLogger(__name__)


def resolve_representation(
    component: common.Component, concept: common.Concept
) -> common.Representation:
    """Effective representation of `component`, defaulting to untyped string.

    The local representation of the component overrides the core representation of the concept
    it identifies, per `ComponentType`: the concept is the fallback "if a representation
    (LocalRepresentation) is not supplied".
    """

    if component.local_representation is not None:
        return component.local_representation
    if concept.core_representation is not None:
        return concept.core_representation

    _log.debug(
        f"Neither {component=} nor {concept=} declares a representation."
        " Defaulting to untyped string, per SDMX 2.1."
    )
    return string_representation()


def string_representation() -> common.Representation:
    """A new untyped-string representation.

    Returns a fresh object on every call: `sdmx1` model objects are mutable, so components must
    not share one.
    """

    return common.Representation(
        non_enumerated=[common.Facet(value_type=common.FacetValueType.string)]
    )
