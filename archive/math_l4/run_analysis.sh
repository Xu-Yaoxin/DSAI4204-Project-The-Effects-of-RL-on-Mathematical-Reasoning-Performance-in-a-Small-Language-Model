#!/usr/bin/env bash
set -euo pipefail
REVIEW_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_CODE="$(CDPATH= cd -- "$REVIEW_DIR/../../.." && pwd)"
python3 -B -u "$PROJECT_CODE/math_l4_rl_split_v1/scripts/merge_shards.py" \
  --experiment-root "$REVIEW_DIR/experiment"
python3 -B -u "$REVIEW_DIR/experiment/scripts/analyze.py" \
  --base outputs/math_l4_v1_fix1/test_base \
  --rl outputs/math_l4_v1_fix1/test_rl_split_merged \
  --output outputs/math_l4_v1_fix1/main_analysis \
  --bootstrap 2000 --seed 42
python3 -B "$REVIEW_DIR/build_report.py"
python3 -B "$REVIEW_DIR/plot_summary.py"
