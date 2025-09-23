import typing as t
from abc import ABC

from sdmx.model import common

from common.data.base import BaseEntity

IdentifiableArtefactType = t.TypeVar("IdentifiableArtefactType", bound=common.IdentifiableArtefact)
NameableArtefactType = t.TypeVar("NameableArtefactType", bound=common.NameableArtefact)


def _extract_short_urn(artefact: common.IdentifiableArtefact) -> str:
    try:
        if isinstance(artefact, common.Item) and artefact.parent:
            parent = artefact.parent
            if parent.id and parent.version and parent.maintainer and parent.maintainer.id:
                return f"{parent.maintainer.id}:{parent.id}({parent.version}).{artefact.id}"
            else:
                return artefact.id
        elif (
            isinstance(artefact, common.MaintainableArtefact)
            and isinstance(artefact, common.VersionableArtefact)
            and artefact.maintainer
            and artefact.maintainer.id
            and artefact.version
        ):
            return f"{artefact.maintainer.id}:{artefact.id}({artefact.version})"
        else:
            return artefact.id
    except AttributeError:
        return artefact.id


class BaseIdentifiableArtefact(BaseEntity, t.Generic[IdentifiableArtefactType], ABC):
    _artefact: IdentifiableArtefactType
    _short_urn: str

    def __init__(self, artefact: IdentifiableArtefactType):
        super().__init__()
        self._artefact = artefact
        self._short_urn = _extract_short_urn(artefact)

    @property
    def source_id(self) -> str:
        return self._artefact.id

    @property
    def short_urn(self) -> str:
        return self._short_urn


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
        return self._artefact.name.localized_default(self._locale)

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
