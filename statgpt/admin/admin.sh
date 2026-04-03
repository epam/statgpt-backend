#!/bin/sh

# IMPORTANT: Do not delete this line!
# These settings make the script fail fast and loud on common errors:
# undefined variables and command failures
set -eu

echo "ADMIN_MODE = '${ADMIN_MODE:-}'"

case "${ADMIN_MODE:-}" in

  APP)
    uvicorn "statgpt.admin.app:app" --host "0.0.0.0" --port 8000 --lifespan on
    ;;

  ALEMBIC_UPGRADE)
    alembic upgrade head
    ;;

  FIX_STATUSES)
    python -m statgpt.admin.fix_statuses
    ;;

  INIT)
    alembic upgrade head
    python -m statgpt.admin.fix_statuses
    ;;

  AUTO_UPDATE)
    python -m statgpt.admin.auto_update
    ;;

  *)
    echo "Unknown ADMIN_MODE = '${ADMIN_MODE:-}'. Possible values:"
    echo "  APP - start the admin application"
    echo "  ALEMBIC_UPGRADE - run alembic migrations to upgrade the database"
    echo "  FIX_STATUSES - fix inconsistent statuses in the database"
    echo "  INIT - run alembic migrations and fix inconsistent statuses"
    echo "  AUTO_UPDATE - run batch auto-update for all eligible channels"
    exit 1
    ;;
esac
