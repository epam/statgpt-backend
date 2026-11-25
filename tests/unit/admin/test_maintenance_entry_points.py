import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OIDC_SETTINGS_MODULE = "statgpt.admin.settings.oidc_auth"


def _env_without_oidc_config() -> dict[str, str]:
    # Maintenance runs in the init container and CronJobs, which get no OIDC configuration.
    # `tests/unit/conftest.py` disables OIDC for this process, so the check runs in a fresh interpreter.
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OIDC_", "ADMIN_ROLES_", "ADMIN_SCOPE_"))
    }


@pytest.mark.parametrize("module", ["statgpt.admin.fix_statuses", "statgpt.admin.auto_update"])
def test_maintenance_entry_point_does_not_load_oidc_settings(module: str) -> None:
    code = f"import sys, {module}; sys.exit({_OIDC_SETTINGS_MODULE!r} in sys.modules)"

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=_env_without_oidc_config(),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"{module} must import without OIDC configuration and must not load "
        f"{_OIDC_SETTINGS_MODULE}.\n{result.stderr}"
    )
