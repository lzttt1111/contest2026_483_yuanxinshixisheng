# Final acceptance commands

The wrapper runs exactly four model surfaces serially: baseline local, merged
local, baseline cloud batch, and merged cloud batch. Each surface refuses a
non-empty destination and retains `_evidence/surface_manifest.json`, logs,
resource samples, partial responses, and its final result on failure.

```bash
bash scripts/acceptance/run_four_surfaces.sh
```

The merged local surface preflights and compares its six formal DOCX against
`00_runtime_assets/frozen_formal_word_baseline` and its relative SHA manifest.
Baseline Word is report-only; cloud surfaces neither generate nor gate Word.

The wrapper always uses the shared main environment at `../.venv/bin/python`
relative to the delivery suite. It never creates or synchronizes another
environment.

The baseline environment is exposed only as a temporary ignored `.venv`
symlink during the baseline local command. The harness removes that link in a
`finally` path and re-proves baseline HEAD, tracked state, untracked state, and
ignored state before accepting the surface.

The final comparison files are written to
`04_acceptance/comparisons/deep_comparison.json` and
`04_acceptance/comparisons/deep_comparison.md`. XLSX is forbidden by the final
contract; the comparator records CSV cells and reports every unexpected XLSX.
