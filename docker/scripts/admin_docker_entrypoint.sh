#!/bin/sh
set -eu

. ./.venv/bin/activate

# If no args passed to `docker run`, then run the admin service
if [ $# -lt 1 ]; then
  exec sh statgpt/admin/admin.sh
fi

# Otherwise, run the user's command, for example a `sh` shell to explore the container
exec "$@"
