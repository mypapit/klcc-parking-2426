#!/usr/bin/env bash
# runs after the E1 grid process exits; rescoring is inference-only, no retraining
set -euo pipefail
cd /edata/opt/klcc-experiment/klcc_ad
PY=../.venv/bin/python
# wait for the E1 runner and any run_one child to finish
while pgrep -f 'scripts/run_experiments.py --exp E1' >/dev/null; do
  sleep 30
done
echo "$(date '+%F %T') E1 finished; starting full-copy rescoring"
$PY scripts/run_injection_copies.py --exp E3
echo "$(date '+%F %T') full-copy rescoring done"
$PY scripts/make_summary.py
