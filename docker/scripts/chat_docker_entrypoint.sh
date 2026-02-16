#!/bin/sh
set -eu

export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# If no args passed to `docker run`, then we assume the user is calling the service
if [ $# -lt 1 ]; then
  exec uvicorn statgpt.app.app:app \
    --host 0.0.0.0 \
    --port 5000 \
    --lifespan on \
    --timeout-keep-alive "${TIMEOUT_KEEP_ALIVE:-5}"
fi

# Otherwise, we assume the user wants to run his own process, for example a `bash` shell to explore the container
exec "$@"
