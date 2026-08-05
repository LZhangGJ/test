#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m tamoe.smoke \
  --config "${project_root}/configs/m0_smoke.json" \
  --output-root "${project_root}/runs" \
  --project-root "${project_root}"
