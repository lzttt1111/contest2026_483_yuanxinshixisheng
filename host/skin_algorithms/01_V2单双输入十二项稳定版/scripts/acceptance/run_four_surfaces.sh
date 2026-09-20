#!/usr/bin/env bash
set -euo pipefail

TOOL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
SUITE_ROOT=$(cd -- "$TOOL_ROOT/.." && pwd)
BASELINE_ROOT="$SUITE_ROOT/01_dev_baseline"
MERGED_ROOT="$SUITE_ROOT/02_dev_merged"
INPUT_MANIFEST="$SUITE_ROOT/03_fixed_inputs/FIXED_INPUTS_MANIFEST.json"
VENDOR_MANIFEST="$SUITE_ROOT/00_runtime_assets/vendor_skin/VENDOR_RUNTIME_MANIFEST.json"
FROZEN_WORD_ROOT="$SUITE_ROOT/00_runtime_assets/frozen_formal_word_baseline"
ACCEPTANCE_ROOT="$SUITE_ROOT/04_acceptance"
COMPARISON_ROOT="$ACCEPTANCE_ROOT/comparisons"
SURFACE_RUNNER="$MERGED_ROOT/scripts/acceptance/run_acceptance_surface.py"
ACCEPTANCE_PYTHON="$SUITE_ROOT/../.venv/bin/python"

test -x "$ACCEPTANCE_PYTHON"

COMMON=(
  --python "$ACCEPTANCE_PYTHON"
  --inputs-manifest "$INPUT_MANIFEST"
  --vendor-manifest "$VENDOR_MANIFEST"
)

mkdir -p "$COMPARISON_ROOT"
if find "$COMPARISON_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "comparison output directory must be empty" >&2
  exit 2
fi

# Surface 1: baseline local, one resident cold start, ordered 25/09/23.
"$ACCEPTANCE_PYTHON" -B "$SURFACE_RUNNER" \
  --profile baseline --surface run --project-root "$BASELINE_ROOT" \
  --output "$ACCEPTANCE_ROOT/baseline/run" "${COMMON[@]}"

# Surface 2: merged local, one resident cold start, ordered 25/09/23, dual Word on.
"$ACCEPTANCE_PYTHON" -B "$SURFACE_RUNNER" \
  --profile merged --surface run --project-root "$MERGED_ROOT" \
  --formal-baseline-root "$FROZEN_WORD_ROOT" \
  --output "$ACCEPTANCE_ROOT/merged/run" "${COMMON[@]}"

# Surface 3: baseline cloud batch, three persistent source-service PIDs once.
"$ACCEPTANCE_PYTHON" -B "$SURFACE_RUNNER" \
  --profile baseline --surface cloud --project-root "$BASELINE_ROOT" \
  --output "$ACCEPTANCE_ROOT/baseline/cloud" "${COMMON[@]}"

# Surface 4: merged cloud batch, three persistent source-service PIDs once.
"$ACCEPTANCE_PYTHON" -B "$SURFACE_RUNNER" \
  --profile merged --surface cloud --project-root "$MERGED_ROOT" \
  --output "$ACCEPTANCE_ROOT/merged/cloud" "${COMMON[@]}"

"$ACCEPTANCE_PYTHON" -B "$MERGED_ROOT/scripts/acceptance/deep_compare.py" \
  --baseline-root "$BASELINE_ROOT" \
  --merged-root "$MERGED_ROOT" \
  --baseline-run "$ACCEPTANCE_ROOT/baseline/run/results" \
  --merged-run "$ACCEPTANCE_ROOT/merged/run/results" \
  --baseline-cloud "$ACCEPTANCE_ROOT/baseline/cloud/results" \
  --merged-cloud "$ACCEPTANCE_ROOT/merged/cloud/results" \
  --formal-baseline-root "$FROZEN_WORD_ROOT" \
  --json-output "$COMPARISON_ROOT/deep_comparison.json" \
  --markdown-output "$COMPARISON_ROOT/deep_comparison.md"
