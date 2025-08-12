from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func


class IdMixin:
    id: Mapped[int] = mapped_column(primary_key=True)


class DateMixin:

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        # server_onupdate=func.now() -- doesn't work for Postgres
    )
