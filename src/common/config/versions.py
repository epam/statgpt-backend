import os


class Versions:
    GIT_COMMIT = os.getenv('GIT_COMMIT', 'unknown')

    # Please update this version when you create a new alembic revision.
    # Needed because alembic folder exist only in the admin_portal package.
    # (statgpt Dockerfile doesn't copy admin_portal package to the container)
    ALEMBIC_TARGET_VERSION = 'f6f95fda8420'
