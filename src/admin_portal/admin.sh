#!/bin/bash

echo "ADMIN_MODE = '$ADMIN_MODE'"

case $ADMIN_MODE in

  APP)
    uvicorn "admin_portal.app:app" --host "0.0.0.0" --port 8000 --lifespan on
    ;;

  ALEMBIC_UPGRADE)
    alembic upgrade head
    ;;

  FIX_STATUSES)
    python -m admin_portal.fix_statuses
    ;;

  INIT)
    alembic upgrade head
    python -m admin_portal.fix_statuses
    ;;

  *)
    echo "Unknown ADMIN_MODE = '$ADMIN_MODE'. Possible values:"
    echo "  APP - start the admin portal application"
    echo "  ALEMBIC_UPGRADE - run alembic migrations to upgrade the database"
    echo "  FIX_STATUSES - fix inconsistent statuses in the database"
    echo "  INIT - run alembic migrations and fix inconsistent statuses"
    exit 1
    ;;
esac
