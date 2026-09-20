#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
group="${LAB_RUNTIME_GROUP:-cpu}"
if [[ "$group" != "cpu" && "$group" != "cuda" ]]; then
  echo "LAB_RUNTIME_GROUP must be cpu or cuda" >&2
  exit 2
fi
exec uv run --frozen --no-default-groups --group "$group" python tools/run_server.py
