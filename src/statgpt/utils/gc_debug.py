"""Garbage collection debugging utilities."""

import gc
import weakref
from typing import Any

from common.config import multiline_logger as logger


class GCDebugger:
    """Track specific objects and their lifecycle."""

    def __init__(self):
        self.tracked_objects = {}

    def track_object(self, obj: Any, name: str):
        """Track an object with a weak reference."""

        def on_delete(ref):
            logger.info(f"GC: Object '{name}' (type={obj_type}) has been garbage collected")

        obj_type = type(obj).__name__
        weak_ref = weakref.ref(obj, on_delete)
        self.tracked_objects[name] = weak_ref
        logger.info(f"GC: Now tracking object '{name}' (type={obj_type}, id={id(obj)})")

    def check_tracked_objects(self):
        """Check which tracked objects are still alive."""
        alive = []
        dead = []

        for name, weak_ref in self.tracked_objects.items():
            if weak_ref() is not None:
                obj = weak_ref()
                alive.append((name, type(obj).__name__, id(obj)))
            else:
                dead.append(name)

        if alive:
            logger.info("GC: Still alive objects:")
            for name, obj_type, obj_id in alive:
                logger.info(f"  - {name} (type={obj_type}, id={obj_id})")

        if dead:
            logger.info(f"GC: Collected objects: {dead}")

        return alive, dead

    def find_referrers(self, obj_name: str, limit: int = 5):
        """Find who's holding references to a tracked object."""
        weak_ref = self.tracked_objects.get(obj_name)
        if weak_ref is None:
            return f"Object '{obj_name}' is not tracked"

        obj = weak_ref()
        if obj is None:
            return f"Object '{obj_name}' has been garbage collected"

        referrers = gc.get_referrers(obj)
        logger.info(f"GC: Object '{obj_name}' has {len(referrers)} referrers:")

        for i, ref in enumerate(referrers[:limit]):
            ref_type = type(ref).__name__
            if ref_type == 'frame':
                continue  # Skip frame objects

            logger.info(f"  [{i}] Type: {ref_type}")

            # Try to get more info
            if hasattr(ref, '__name__'):
                logger.info(f"      Name: {ref.__name__}")
            if hasattr(ref, '__class__'):
                logger.info(f"      Class: {ref.__class__.__name__}")
            if isinstance(ref, dict):
                keys = list(ref.keys())[:3]
                logger.info(f"      Dict keys: {keys}")

        return f"Found {len(referrers)} referrers for '{obj_name}'"


# Global GC debugger instance
gc_debugger = GCDebugger()


def log_gc_stats():
    """Log garbage collection statistics."""
    stats = gc.get_stats()
    for i, generation in enumerate(stats):
        logger.info(
            f"GC Generation {i}: collections={generation['collections']}, "
            f"collected={generation['collected']}, uncollectable={generation['uncollectable']}"
        )

    # Force collection and report
    logger.info("GC: Forcing collection...")
    collected = gc.collect()
    logger.info(f"GC: Collected {collected} objects")

    # Check for uncollectable objects
    if gc.garbage:
        logger.warning(f"GC: Found {len(gc.garbage)} uncollectable objects!")
        for obj in gc.garbage[:5]:
            logger.warning(f"  - Type: {type(obj).__name__}, ID: {id(obj)}")
