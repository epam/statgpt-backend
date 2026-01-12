import typing as t
from abc import ABC

from sdmx.model import common

from statgpt.common.data.base import BaseEntity

IdentifiableArtefactType = t.TypeVar("IdentifiableArtefactType", bound=common.IdentifiableArtefact)
NameableArtefactType = t.TypeVar("NameableArtefactType", bound=common.NameableArtefact)


class FullUrn(t.NamedTuple):
    agency_id: str | None
    resource_id: str | None
    version: str | None
    item_id: str | None

    @property
    def short_urn(self) -> str:
        if self.agency_id and self.resource_id and self.version:
            urn = f"{self.agency_id}:{self.resource_id}({self.version})"
            if self.item_id:
                urn += f".{self.item_id}"
        elif self.item_id:
            urn = self.item_id
        elif self.resource_id:
            urn = self.resource_id
        else:
            raise ValueError("Insufficient data to construct URN.")

        return urn

    @classmethod
    def from_artefact(cls, artefact: common.IdentifiableArtefact) -> t.Self:
        try:
            if isinstance(artefact, common.Item):
                if artefact.parent:
                    parent = artefact.parent
                    return cls(
                        agency_id=parent.maintainer.id if parent.maintainer else None,
                        resource_id=parent.id,
                        version=str(parent.version) if parent.version else None,
                        item_id=artefact.id,
                    )
                else:
                    return cls(agency_id=None, resource_id=None, version=None, item_id=artefact.id)
            elif (
                isinstance(artefact, common.MaintainableArtefact)
                and isinstance(artefact, common.VersionableArtefact)
                and artefact.maintainer
                and artefact.maintainer.id
                and artefact.version
            ):
                return cls(
                    agency_id=artefact.maintainer.id,
                    resource_id=artefact.id,
                    version=str(artefact.version) if artefact.version else None,
                    item_id=None,
                )
            else:
                return cls(agency_id=None, resource_id=artefact.id, version=None, item_id=None)
        except AttributeError:
            return cls(agency_id=None, resource_id=artefact.id, version=None, item_id=None)


class BaseIdentifiableArtefact(BaseEntity, t.Generic[IdentifiableArtefactType], ABC):
    _artefact: IdentifiableArtefactType
    _full_urn: FullUrn
    _short_urn: str

    def __init__(self, artefact: IdentifiableArtefactType):
        super().__init__()
        self._artefact = artefact
        self._full_urn = FullUrn.from_artefact(artefact)
        self._short_urn = self._full_urn.short_urn

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
