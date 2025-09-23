"""Memory debugging endpoints for development."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from common.utils.memory_profiler import memory_profiler

router = APIRouter(prefix="/debug/memory", tags=["debug"])


@router.get("/snapshot")
async def take_memory_snapshot(label: str | None = None):
    """Take a memory snapshot."""
    index = memory_profiler.take_snapshot(label)
    return {"message": "Snapshot taken", "index": index, "label": label}


@router.get("/compare")
async def compare_snapshots(index1: int = -2, index2: int = -1):
    """Compare two memory snapshots."""
    result = memory_profiler.compare_snapshots(index1, index2)
    return PlainTextResponse(content=result)


@router.get("/top")
async def get_top_memory_users(top_n: int = 20):
    """Get top memory users."""
    result = memory_profiler.get_top_memory_users(top_n)
    return PlainTextResponse(content=result)


@router.get("/objects")
async def track_object_counts():
    """Track object count changes."""
    result = memory_profiler.track_object_counts()
    return PlainTextResponse(content=result)


@router.get("/sessions")
async def get_session_info():
    """Get information about active SQLAlchemy sessions."""
    result = memory_profiler.get_session_info()
    return PlainTextResponse(content=result)


@router.get("/class/{class_name}")
async def track_specific_class(class_name: str):
    """Track instances of a specific class."""
    result = memory_profiler.track_specific_class(class_name)
    return PlainTextResponse(content=result)


@router.get("/uncollected")
async def find_uncollected_objects():
    """Find objects that cannot be garbage collected."""
    result = memory_profiler.find_uncollected_objects()
    return PlainTextResponse(content=result)


@router.get("/summary")
async def get_memory_summary():
    """Get comprehensive memory summary."""
    result = memory_profiler.get_summary()
    return PlainTextResponse(content=result)


@router.post("/cleanup")
async def cleanup_memory_profiler():
    """Clean up all temporary snapshot files and free disk space."""
    memory_profiler.cleanup()
    return {"message": "Memory profiler cleaned up successfully"}
