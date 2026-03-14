#!/bin/bash
set -euo pipefail

RUN_SCRIPT="${RUN_SCRIPT:-run_2d_single.sh}"
ZERO_HV_LIST="${ZERO_HV_LIST:-0 1}"
NEGATE_MODE_LIST="${NEGATE_MODE_LIST:-both}"
STRATEGY_LIST="${STRATEGY_LIST:-random ucb ellipse_fast ellipse_directions}"

for ZERO_HV in $ZERO_HV_LIST; do
  for NEGATE_MODE in $NEGATE_MODE_LIST; do
    for STRATEGY in $STRATEGY_LIST; do
      JOB_TAG="2d_${STRATEGY}_${NEGATE_MODE}_c${ZERO_HV}"
      sbatch \
        --job-name="$JOB_TAG" \
        --output="logs/${JOB_TAG}_%j.out" \
        --error="logs/${JOB_TAG}_%j.err" \
        --export=ALL,ZERO_HV="$ZERO_HV",NEGATE_MODE="$NEGATE_MODE",STRATEGY="$STRATEGY" \
        "$RUN_SCRIPT"
    done
  done
done
