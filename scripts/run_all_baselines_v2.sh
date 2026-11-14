#!/usr/bin/env bash
# Fix 6: B1-B4 with the 5-copy protocol on Test A and Test B; B3 with 5 seeds.
set -uo pipefail
cd /edata/opt/klcc-experiment/klcc_ad
PY=../.venv/bin/python
for L in 6 18 36; do
  for B in B1 B2 B4; do
    echo "=== $B L=$L s0 ==="
    $PY scripts/run_baselines_v2.py --baseline $B --length $L --seed 0 || echo "FAILED $B L$L s0"
  done
  for S in 0 1 2 3 4; do
    echo "=== B3 L=$L s$S ==="
    $PY scripts/run_baselines_v2.py --baseline B3 --length $L --seed $S || echo "FAILED B3 L$L s$S"
  done
done
echo "ALL BASELINES DONE"
