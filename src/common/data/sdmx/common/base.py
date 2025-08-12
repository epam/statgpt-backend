import typing as t
from abc import ABC

from sdmx.model import common

from common.data.base import BaseEntity

from .urn import Urn, UrnParser

IdentifiableArtefactType = t.TypeVar("IdentifiableArtefactType", bound=common.IdentifiableArtefact)
NameableArtefactType = t.TypeVar("NameableArtefactType", bound=common.NameableArtefact)


class BaseIdentifiableArtefact(BaseEntity, t.Generic[IdentifiableArtefactType], ABC):
    _artefact: IdentifiableArtefactType
    _urn: Urn

    def __init__(self, artefact: IdentifiableArtefactType):
        super().__init__()
        self._artefact = artefact
        urn_parser = UrnParser.create_default()
        if isinstance(artefact, common.Item) and artefact.parent and artefact.parent.urn:
            # NOTE: here, for each code list's item, we parse URN of the parent dataflow.
            # probably it's better to pass this URN as a param to avoid invoking parsing each time,
            # however there is no bottleneck right now.
            self._urn = urn_parser.parse(artefact.parent.urn)
            self._urn.item_id = artefact.id
        elif not artefact.urn:
            self._urn = artefact.id  # type: ignore
        else:
            self._urn = urn_parser.parse(artefact.urn)

    @property
    def source_id(self) -> str:
        return self._urn.get_urn()

    @property
    def urn(self) -> Urn:
        return self._urn

    @property
    def short_urn(self) -> str:
        return self._urn.get_short_urn()


class BaseNameableArtefact(
    BaseIdentifiableArtefact[NameableArtefactType],
    t.Generic[NameableArtefactType],
    ABC,
):
    _locale: str

    def __init__(self, artefact: NameableArtefactType, locale: str):
        BaseEntity.__init__(self)  # TODO: looks like a redundant call
        BaseIdentifiableArtefact.__init__(self, artefact)
        self._locale = locale

    @property
    def name(self) -> str:
        return self._artefact.name[self._locale]

    @property
    def description(self) -> t.Optional[str]:
        return self._artefact.description.localizations.get(self._locale)

    def annotation(self, annotation_id: str) -> str | None:
        annotation = next((a for a in self._artefact.annotations if a.id == annotation_id), None)
        if not annotation or not annotation.text:
            return None
        return (
            annotation.text[self._locale]
            if (annotation and annotation.text and self._locale in annotation.text.localizations)
            else None
        )
