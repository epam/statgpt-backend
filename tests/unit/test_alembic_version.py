"""Unit test to verify that the ALEMBIC_TARGET_VERSION matches the latest migration."""

import ast
from pathlib import Path

from src.common.config.versions import Versions


def extract_revision_from_migration(file_path: Path) -> str:
    """Extract the revision ID from an alembic migration file.

    Args:
        file_path: Path to the migration file

    Returns:
        The revision ID string

    Raises:
        ValueError: If revision cannot be found in the file
    """
    with open(file_path, 'r') as f:
        content = f.read()

    tree = ast.parse(content)

    for node in ast.walk(tree):
        # Handle both annotated (revision: str = 'x') and regular (revision = 'x') assignments
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == 'revision':
                if isinstance(node.value, ast.Constant):
                    return node.value.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'revision':
                    if isinstance(node.value, ast.Constant):
                        return node.value.value

    raise ValueError(f"Could not find revision in {file_path}")


def get_latest_migration_revision() -> str:
    """Get the revision ID of the latest alembic migration.

    Returns:
        The revision ID of the most recent migration file

    Raises:
        FileNotFoundError: If no migration files are found
    """
    base_dir = Path(__file__).parent.parent.parent
    versions_dir = base_dir / "src" / "admin_portal" / "alembic" / "versions"

    if not versions_dir.exists():
        raise FileNotFoundError(f"Alembic versions directory not found: {versions_dir}")

    migration_files = sorted(versions_dir.glob("*.py"))

    if not migration_files:
        raise FileNotFoundError(f"No migration files found in {versions_dir}")

    latest_migration = migration_files[-1]  # Files sorted by timestamp in filename
    return extract_revision_from_migration(latest_migration)


class TestAlembicVersion:
    """Test that the ALEMBIC_TARGET_VERSION matches the latest migration."""

    def test_version_matches_latest_migration(self):
        """Verify that ALEMBIC_TARGET_VERSION equals the latest migration revision."""
        configured_version = Versions.ALEMBIC_TARGET_VERSION
        latest_revision = get_latest_migration_revision()

        assert configured_version == latest_revision, (
            f"ALEMBIC_TARGET_VERSION ({configured_version}) does not match "
            f"the latest migration revision ({latest_revision}). "
            f"Please update ALEMBIC_TARGET_VERSION in src/common/config/versions.py"
        )

    def test_configured_version_not_unknown(self):
        """Verify that ALEMBIC_TARGET_VERSION is not set to a placeholder value."""
        configured_version = Versions.ALEMBIC_TARGET_VERSION

        assert configured_version != 'unknown', (
            "ALEMBIC_TARGET_VERSION should not be 'unknown'. "
            "Please set it to the latest migration revision."
        )

        assert configured_version != '', (
            "ALEMBIC_TARGET_VERSION should not be empty. "
            "Please set it to the latest migration revision."
        )

    def test_latest_migration_extraction(self):
        """Test that we can successfully extract a revision from migration files."""
        latest_revision = get_latest_migration_revision()

        assert isinstance(latest_revision, str)
        assert len(latest_revision) == 12
        assert all(c in '0123456789abcdef' for c in latest_revision)
