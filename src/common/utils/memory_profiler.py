import atexit
import gc
import linecache
import os
import pickle
import shutil
import tempfile
import tracemalloc
from collections import defaultdict
from datetime import datetime
from typing import Optional

from common.config import multiline_logger as logger


class MemoryProfiler:
    """Memory profiler to track memory leaks in the application."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # Create a temporary directory for snapshots
            self.temp_dir = tempfile.mkdtemp(prefix="memory_snapshots_")
            self.snapshot_files = []  # Store tempfile objects
            self.snapshot_metadata = []  # Only store metadata in memory
            self.object_counts = defaultdict(int)
            self.weak_refs = []
            self._initialized = True
            self.start_profiling()
            # Register cleanup on exit
            atexit.register(self.cleanup)

    def start_profiling(self):
        """Start memory profiling with tracemalloc."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(3)  # Reduce frames to minimize overhead
            logger.info("Memory profiling started")

    def take_snapshot(self, label: str | None = None):
        """Take a memory snapshot and save to disk."""
        gc.collect()  # Force garbage collection before snapshot

        snapshot = tracemalloc.take_snapshot()
        timestamp = datetime.now().isoformat()

        # Create a temporary file for the snapshot
        snapshot_id = len(self.snapshot_metadata)

        # Create named temporary file that won't be auto-deleted
        temp_file = tempfile.NamedTemporaryFile(
            mode='wb',
            prefix=f"snapshot_{snapshot_id}_",
            suffix=".pkl",
            dir=self.temp_dir,
            delete=False,
        )

        try:
            pickle.dump(snapshot, temp_file)
            temp_file.flush()

            # Store the file object and metadata
            self.snapshot_files.append(temp_file)
            self.snapshot_metadata.append(
                {
                    'timestamp': timestamp,
                    'label': label or f"snapshot_{snapshot_id}",
                    'file_path': temp_file.name,
                    'id': snapshot_id,
                }
            )

            logger.info(f"Memory snapshot taken and saved: {label or timestamp}")
        finally:
            temp_file.close()

        # Auto-cleanup old snapshots to prevent disk fill
        if len(self.snapshot_metadata) > 20:
            self.cleanup_old_snapshots(keep_last=10)

        return snapshot_id

    def _load_snapshot(self, index: int) -> Optional[tracemalloc.Snapshot]:
        """Load a snapshot from disk."""
        if index < 0:
            index = len(self.snapshot_metadata) + index

        if 0 <= index < len(self.snapshot_metadata):
            snapshot_path = self.snapshot_metadata[index]['file_path']
            try:
                with open(snapshot_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load snapshot {index}: {e}")
                return None
        return None

    def compare_snapshots(self, index1: int = -2, index2: int = -1, top_n: int = 20):
        """Compare two snapshots and return top memory consumers."""
        if len(self.snapshot_metadata) < 2:
            return "Need at least 2 snapshots to compare"

        snap1 = self._load_snapshot(index1)
        snap2 = self._load_snapshot(index2)

        if not snap1 or not snap2:
            return "Failed to load snapshots for comparison"

        top_stats = snap2.compare_to(snap1, 'lineno')

        result = []
        # Adjust indices for metadata access
        idx1 = index1 if index1 >= 0 else len(self.snapshot_metadata) + index1
        idx2 = index2 if index2 >= 0 else len(self.snapshot_metadata) + index2

        result.append(
            f"\n=== Memory Changes from {self.snapshot_metadata[idx1]['label']} to {self.snapshot_metadata[idx2]['label']} ===\n"
        )

        for stat in top_stats[:top_n]:
            if stat.size_diff > 0:
                result.append(f"{stat.traceback.format()[0]}")
                result.append(f"  Size diff: +{stat.size_diff / 1024 / 1024:.2f} MB")
                result.append(f"  Count diff: +{stat.count_diff}")
                result.append("")

        return "\n".join(result)

    def get_top_memory_users(self, top_n: int = 20):
        """Get current top memory users."""
        if not self.snapshot_metadata:
            self.take_snapshot("initial")

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        result = []
        result.append("\n=== Top Memory Users ===\n")

        for stat in top_stats[:top_n]:
            frame = stat.traceback[0]
            result.append(f"{frame.filename}:{frame.lineno}")
            result.append(f"  Size: {stat.size / 1024 / 1024:.2f} MB")
            result.append(f"  Count: {stat.count}")

            # Try to get the actual line of code
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                result.append(f"  Code: {line}")
            result.append("")

        return "\n".join(result)

    def track_object_counts(self):
        """Track counts of different object types."""
        gc.collect()

        counts = defaultdict(int)
        for obj in gc.get_objects():
            counts[type(obj).__name__] += 1

        # Find objects that increased
        increases = []
        for obj_type, count in counts.items():
            old_count = self.object_counts.get(obj_type, 0)
            if count > old_count:
                increases.append((obj_type, count - old_count, count))

        self.object_counts = counts

        if increases:
            result = ["\n=== Object Count Increases ===\n"]
            for obj_type, increase, total in sorted(increases, key=lambda x: x[1], reverse=True)[
                :20
            ]:
                result.append(f"{obj_type}: +{increase} (total: {total})")
            return "\n".join(result)
        return "No significant object count increases"

    def find_uncollected_objects(self):
        """Find objects that cannot be garbage collected."""
        gc.collect()
        uncollected = gc.collect()

        if uncollected > 0:
            result = [f"\n=== Found {uncollected} uncollectable objects ===\n"]

            for obj in gc.garbage[:10]:  # Show first 10
                result.append(f"Type: {type(obj).__name__}")
                result.append(f"ID: {id(obj)}")

                # Try to get more info about the object
                if hasattr(obj, '__dict__'):
                    attrs = list(obj.__dict__.keys())[:5]
                    result.append(f"Attributes: {attrs}")

                result.append("")

            return "\n".join(result)
        return "No uncollectable objects found"

    def track_specific_class(self, class_name: str):
        """Track instances of a specific class."""
        gc.collect()

        instances = []
        for obj in gc.get_objects():
            if type(obj).__name__ == class_name:
                instances.append(obj)

        result = [f"\n=== {class_name} Instances: {len(instances)} ===\n"]

        for i, obj in enumerate(instances[:10]):  # Show first 10
            result.append(f"Instance #{i}:")
            result.append(f"  ID: {id(obj)}")
            result.append(f"  Referrers: {len(gc.get_referrers(obj))}")

            # Show who's holding references
            referrers = gc.get_referrers(obj)
            for ref in referrers[:3]:  # Show first 3 referrers
                ref_type = type(ref).__name__
                if ref_type != 'frame':  # Skip frame objects
                    result.append(f"    Referenced by: {ref_type}")

            result.append("")

        return "\n".join(result)

    def get_session_info(self):
        """Get information about SQLAlchemy sessions."""
        gc.collect()

        sessions = []
        for obj in gc.get_objects():
            if 'AsyncSession' in type(obj).__name__:
                sessions.append(obj)

        result = [f"\n=== Active AsyncSessions: {len(sessions)} ===\n"]

        for i, session in enumerate(sessions[:10]):
            result.append(f"Session #{i}:")
            result.append(f"  Type: {type(session).__name__}")
            result.append(f"  ID: {id(session)}")

            # Check identity map size
            if hasattr(session, 'identity_map'):
                if hasattr(session.identity_map, '_dict'):
                    result.append(f"  Identity map size: {len(session.identity_map._dict)}")
                elif hasattr(session.identity_map, '__len__'):
                    result.append(f"  Identity map size: {len(session.identity_map)}")

            result.append(f"  Referrers: {len(gc.get_referrers(session))}")
            result.append("")

        return "\n".join(result)

    def get_summary(self):
        """Get a comprehensive memory summary."""
        result = []

        result.append(self.get_top_memory_users(10))
        result.append(self.track_object_counts())
        result.append(self.find_uncollected_objects())
        result.append(self.get_session_info())
        result.append(self.track_specific_class('ChannelServiceFacade'))
        result.append(self.track_specific_class('VectorStore'))
        result.append(self.track_specific_class('PgVectorStore'))

        if len(self.snapshot_metadata) >= 2:
            result.append(self.compare_snapshots())

        return "\n".join(result)

    def cleanup_old_snapshots(self, keep_last: int = 10):
        """Remove old snapshot files to save disk space."""
        if len(self.snapshot_metadata) > keep_last:
            to_remove = len(self.snapshot_metadata) - keep_last
            for i in range(to_remove):
                snapshot_path = self.snapshot_metadata[i]['file_path']
                try:
                    if os.path.exists(snapshot_path):
                        os.unlink(snapshot_path)
                        logger.info(f"Removed old snapshot: {os.path.basename(snapshot_path)}")
                except Exception as e:
                    logger.error(f"Failed to remove snapshot {snapshot_path}: {e}")

            # Update metadata and file list
            self.snapshot_metadata = self.snapshot_metadata[to_remove:]
            self.snapshot_files = self.snapshot_files[to_remove:]

    def cleanup(self):
        """Clean up all temporary files and directories."""
        try:
            # Remove all snapshot files
            for metadata in self.snapshot_metadata:
                snapshot_path = metadata['file_path']
                if os.path.exists(snapshot_path):
                    try:
                        os.unlink(snapshot_path)
                    except Exception as e:
                        logger.error(f"Failed to remove snapshot {snapshot_path}: {e}")

            # Remove the temporary directory
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global profiler instance
memory_profiler = MemoryProfiler()
