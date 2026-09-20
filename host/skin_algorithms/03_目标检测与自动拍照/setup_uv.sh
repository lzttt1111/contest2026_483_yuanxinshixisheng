#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
group="${1:-cpu}"
if [[ "$group" != "cpu" && "$group" != "cuda" ]]; then
  echo "usage: bash setup_uv.sh [cpu|cuda]" >&2
  exit 2
fi
uv sync --frozen --no-default-groups --group "$group"
echo "environment ready: $group"

