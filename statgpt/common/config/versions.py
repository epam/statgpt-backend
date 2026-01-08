import os


class Versions:
    GIT_COMMIT = os.getenv('GIT_COMMIT', 'unknown')

    # Please update this version when you create a new alembic revision.
    # Needed because alembic folder exist only in the statgpt.admin package.
    # (statgpt Dockerfile doesn't copy statgpt.admin package to the container)
    ALEMBIC_TARGET_VERSION = '109220bfa072'
