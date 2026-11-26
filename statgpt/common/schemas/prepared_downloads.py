from uuid import UUID

from pydantic import AwareDatetime, BaseModel

from .base import BaseYamlModel


class PreparedDownloadResponse(BaseYamlModel):
    id: UUID
    expires_at: AwareDatetime


class PreparedDownloadErrorResponse(BaseModel):
    detail: str
