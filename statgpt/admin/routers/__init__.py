from fastapi import APIRouter

from .audit_log import router as audit_log_router
from .channel import router as channel_router
from .data_source import router as data_source_router
from .dataset import router as dataset_router
from .glossary_of_terms import channel_terms_router, terms_router
from .health import router as health_check

router = APIRouter(prefix="/admin/api/v1")

channel_router.include_router(channel_terms_router)

for r in (
    channel_router,
    data_source_router,
    dataset_router,
    terms_router,
    audit_log_router,
    health_check,
):
    router.include_router(r)
