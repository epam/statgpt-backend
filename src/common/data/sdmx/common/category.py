import typing as t
from abc import ABC

from langchain_core.documents import Document
from sdmx.model import common as sdmx_common

from common.data.base import Category, DimensionCategory

from .base import BaseNameableArtefact
from .config import FixedItem
from .constants import SdmxConstants


class CodeCategory(Category, BaseNameableArtefact[sdmx_common.Code]):

    def __init__(self, code: sdmx_common.Code, locale: str):
        Category.__init__(self)
        BaseNameableArtefact.__init__(self, code, locale)

    @property
    def entity_id(self) -> str:
        return self._artefact.id

    @property
    def query_id(self) -> str:
        return self._artefact.id

    def get_document_content(self, include_description: bool = False, **kwargs) -> str:
        content = f"id: {self._artefact.id}, name: {self.name}"
        if self.description and include_description:
            content += f", description: {self.description}"

        return content

    def get_document_metadata(self, **kwargs) -> dict:
        return {
            SdmxConstants.METADATA_CODE_URN: self._artefact.urn,
            SdmxConstants.METADATA_CODE_ID: self._artefact.id,
            SdmxConstants.METADATA_CODE_NAME: self.name,
            SdmxConstants.METADATA_CODE_DESCRIPTION: self.description,
            SdmxConstants.METADATA_CODE_LOCALE: self._locale,
        }

    @classmethod
    def from_document(cls, document: Document) -> 'CodeCategory':
        code = sdmx_common.Code(
            id=document.metadata[SdmxConstants.METADATA_CODE_ID],
            urn=document.metadata[SdmxConstants.METADATA_CODE_URN],
            name=document.metadata[SdmxConstants.METADATA_CODE_NAME],
            description=document.metadata.get(SdmxConstants.METADATA_CODE_DESCRIPTION, ""),
        )
        locale = document.metadata[SdmxConstants.METADATA_CODE_LOCALE]
        return cls(code=code, locale=locale)


class SdmxDimensionCategory(DimensionCategory, ABC):
    """
    Represents a SDMX dimension category. It is used to represent a dimension in SDMX data structures.
    """

    _dimension_id: str
    _dimension_name: str
    _dimension_alias: str | None

    def __init__(
        self,
        dimension_id: str,
        dimension_name: str,
        dimension_alias: str | None = None,
    ):
        self._dimension_id = dimension_id
        self._dimension_name = dimension_name
        self._dimension_alias = dimension_alias

    @property
    def dimension_id(self) -> str:
        return self._dimension_id

    @property
    def dimension_name(self) -> str:
        return self._dimension_name

    @property
    def dimension_alias(self) -> str | None:
        return self._dimension_alias

    def get_document_metadata(self, **kwargs) -> dict:
        metadata = {
            SdmxConstants.METADATA_DIMENSION_ID: self._dimension_id,
            SdmxConstants.METADATA_DIMENSION_NAME: self._dimension_name,
            SdmxConstants.METADATA_DIMENSION_ALIAS: self._dimension_alias,
        }
        return metadata


class DimensionVirtualCodeCategory(SdmxDimensionCategory):
    """
    Represents a virtual code category that is associated with a dimension. Virtual code is not a real code in the
    SDMX codelist.
    """

    fixed_item: FixedItem

    def __init__(
        self,
        fixed_item: FixedItem,
        dimension_id: str,
        dimension_name: str,
        dimension_alias: str | None = None,
    ):
        SdmxDimensionCategory.__init__(self, dimension_id, dimension_name, dimension_alias)
        self.fixed_item = fixed_item

    @property
    def entity_id(self) -> str:
        return self.fixed_item.id

    @property
    def source_id(self) -> str:
        return self.fixed_item.id

    @property
    def name(self) -> str:
        return self.fixed_item.name

    @property
    def description(self) -> t.Optional[str]:
        return self.fixed_item.description

    def get_document_content(self, include_description: bool = False, **kwargs) -> str:
        content = f"id: {self.fixed_item.id}, name: {self.fixed_item.name}"
        if self.fixed_item.description and include_description:
            content += f", description: {self.fixed_item.description}"
        if self._dimension_alias:
            # use alias if available
            return f"{content} ({self._dimension_alias})"
        else:
            return f"{content} ({self._dimension_name})"

    def get_document_metadata(self, **kwargs) -> dict:
        return {
            **SdmxDimensionCategory.get_document_metadata(self, **kwargs),
            SdmxConstants.METADATA_VIRTUAL_DIMENSION_VALUE_ID: self.fixed_item.id,
            SdmxConstants.METADATA_VIRTUAL_DIMENSION_VALUE_NAME: self.fixed_item.name,
            SdmxConstants.METADATA_VIRTUAL_DIMENSION_VALUE_DESCRIPTION: self.fixed_item.description,
        }

    @classmethod
    def from_document(cls, document: Document) -> 'Category':
        dimension_id = document.metadata[SdmxConstants.METADATA_DIMENSION_ID]
        dimension_name = document.metadata[SdmxConstants.METADATA_DIMENSION_NAME]
        dimension_alias = document.metadata.get(SdmxConstants.METADATA_DIMENSION_ALIAS, None)
        fixed_item = FixedItem(
            id=document.metadata[SdmxConstants.METADATA_VIRTUAL_DIMENSION_VALUE_ID],
            name=document.metadata[SdmxConstants.METADATA_VIRTUAL_DIMENSION_VALUE_NAME],
            description=document.metadata.get(
                SdmxConstants.METADATA_VIRTUAL_DIMENSION_VALUE_DESCRIPTION, None
            ),
        )
        return cls(
            fixed_item=fixed_item,
            dimension_id=dimension_id,
            dimension_name=dimension_name,
            dimension_alias=dimension_alias,
        )


class DimensionCodeCategory(SdmxDimensionCategory, CodeCategory):

    def __init__(
        self,
        code: sdmx_common.Code,
        locale: str,
        dimension_id: str,
        dimension_name: str,
        dimension_alias: str | None = None,
    ):
        CodeCategory.__init__(self, code, locale)
        SdmxDimensionCategory.__init__(self, dimension_id, dimension_name, dimension_alias)

    def get_document_content(self, include_description: bool = False, **kwargs) -> str:
        content = CodeCategory.get_document_content(
            self, include_description=include_description, **kwargs
        )
        if self._dimension_alias:
            # use alias if available
            return f"{content} ({self._dimension_alias})"
        else:
            return f"{content} ({self._dimension_name})"

    def get_document_metadata(self, **kwargs) -> dict:
        return {
            **CodeCategory.get_document_metadata(self, **kwargs),
            **SdmxDimensionCategory.get_document_metadata(self, **kwargs),
        }

    @classmethod
    def from_document(cls, document: Document) -> 'DimensionCodeCategory':
        dimension_id = document.metadata[SdmxConstants.METADATA_DIMENSION_ID]
        dimension_name = document.metadata[SdmxConstants.METADATA_DIMENSION_NAME]
        dimension_alias = document.metadata.get(SdmxConstants.METADATA_DIMENSION_ALIAS, None)
        code_category = CodeCategory.from_document(document)
        return cls.from_code_category(code_category, dimension_id, dimension_name, dimension_alias)

    @classmethod
    def from_code_category(
        cls,
        code_category: CodeCategory,
        dimension_id: str,
        dimension_name: str,
        dimension_alias: str | None = None,
    ) -> 'DimensionCodeCategory':
        return cls(
            code=code_category._artefact,
            locale=code_category._locale,
            dimension_id=dimension_id,
            dimension_name=dimension_name,
            dimension_alias=dimension_alias,
        )
