import logging
from dataclasses import dataclass
from typing import NoReturn

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError

_log = logging.getLogger(__name__)

# Postgres SQLSTATE for unique_violation. SQLAlchemy's asyncpg adapter copies the
# driver's sqlstate onto the DBAPI error it raises, so matching on it is stable across
# driver versions - unlike matching on the formatted exception message.
_UNIQUE_VIOLATION_SQLSTATE = "23505"


class AdminServiceError(Exception):
    """Base class for admin service errors mapped to HTTP responses in routers."""


def raise_for_integrity_error(e: IntegrityError, conflict_detail: str) -> NoReturn:
    """Translate a DB integrity failure into an actionable HTTP error.

    A unique-constraint violation becomes a 409 carrying `conflict_detail`, instead of
    leaking a driver traceback as a 500. Any other integrity failure stays a 500.
    """
    _log.warning(e)

    if getattr(e.orig, "sqlstate", None) == _UNIQUE_VIOLATION_SQLSTATE:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=conflict_detail)

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unknown db error"
    )


@dataclass(frozen=True)
class BlockingDataset:
    dataset_id: int
    dataset_title: str
    channel_count: int

    @property
    def channels_label(self) -> str:
        suffix = "" if self.channel_count == 1 else "s"
        return f"{self.channel_count} channel{suffix}"

    @property
    def usage(self) -> str:
        return f"'{self.dataset_title}' ({self.channels_label})"


class DatasetInUseError(AdminServiceError):
    def __init__(self, blocking_datasets: list[BlockingDataset]) -> None:
        self.blocking_datasets = blocking_datasets
        super().__init__(f"Dataset(s) still used in channels: {self.usage_summary}")

    @property
    def usage_summary(self) -> str:
        return ", ".join(ds.usage for ds in self.blocking_datasets)
