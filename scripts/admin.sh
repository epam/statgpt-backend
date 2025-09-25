#!/bin/bash

echo "ADMIN_MODE = '$ADMIN_MODE'"

case $ADMIN_MODE in

  APP)
    uvicorn "admin_portal.app:app" --host "0.0.0.0" --port 8000 --lifespan on
    ;;

  ALEMBIC_UPGRADE)
    alembic upgrade head
    ;;

  *)
    echo "Unknown ADMIN_MODE = '$ADMIN_MODE'. Possible values: 'APP' or 'ALEMBIC_UPGRADE'"
    ;;
esac
