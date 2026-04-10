#!/bin/bash
set -euo pipefail

# ===== Core parameters =====
SEED_MODE="${SEED_MODE:-multi}"                       # multi | single
SEED_FILE_LIST="${SEED_FILE_LIST:-seeds/rep0.txt seeds/rep1.txt seeds/rep2.txt}"
SEED_FILE_SINGLE="${SEED_FILE_SINGLE:-seeds/rep0.txt}"
PROJECT="${PROJECT:-default_2d}"
RESULTS_BASE="${RESULTS_BASE:-results/$PROJECT}"
LOGS_BASE="${LOGS_BASE:-logs/$PROJECT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_BASE}"
LOG_ROOT="${LOG_ROOT:-$LOGS_BASE}"

K_LIST="${K_LIST:-1 2 3 4}"
N_REPLICATES="${N_REPLICATES:-1}"

# ===== Experiment matrix =====
ZERO_HV_LIST="${ZERO_HV_LIST:-0 1}"
NEGATE_MODE_LIST="${NEGATE_MODE_LIST:-both}"         # both | none
#STRATEGY_LIST="${STRATEGY_LIST:-random ucb ellipse_fast ellipse_directions}"
STRATEGY_LIST="${STRATEGY_LIST:-ellipse_fast}"

# ===== Runtime/resources =====
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MEM_PER_JOB="${MEM_PER_JOB:-220G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
PARTITION="${PARTITION:-plgrid-gpu-gh200}"
ACCOUNT="${ACCOUNT:-plgsonata19-gpu-gh200}"
GRES="${GRES:-gpu:1}"

# ===== Environment/data =====
CONDA_INIT="${CONDA_INIT:-/net/storage/pr3/plgrid/plggsanodrugs/miniconda-arm/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-savi-arm}"

DATA_FILE="${DATA_FILE:-data/savi_data.parquet}"
PROPERTY_COLS="${PROPERTY_COLS:-score_3GVB score_6D6P}"
NEGATE_COLS_DEFAULT="${NEGATE_COLS_DEFAULT:-score_3GVB score_6D6P}"
GLOBAL_PARETO_FILE="${GLOBAL_PARETO_FILE:-$DATA_FILE}"
FINGERPRINT_COL="${FINGERPRINT_COL:-X_ecfp_2}"
DEVICE="${DEVICE:-cuda}"

SEED_SIZE="${SEED_SIZE:-100}"
BATCH_SIZE="${BATCH_SIZE:-100}"
N_ITERATIONS="${N_ITERATIONS:-20}"
UCB_INCLUDE_K0="${UCB_INCLUDE_K0:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ===== W&B =====
WANDB_ENABLE="${WANDB_ENABLE:-1}"                    # 0 | 1
WANDB_PROJECT="${WANDB_PROJECT:-mdg-helios-2d}"
WANDB_ENTITY="${WANDB_ENTITY:-jklimczak-sano}"
WANDB_RUN_NAME_PREFIX="${WANDB_RUN_NAME_PREFIX:-2d}"
export WANDB_API_KEY="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"

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT"

seed_items=()
if [[ "$SEED_MODE" == "multi" ]]; then
  read -r -a _seed_arr <<< "$SEED_FILE_LIST"
  if [[ "${#_seed_arr[@]}" -eq 0 ]]; then
    echo "SEED_MODE=multi but SEED_FILE_LIST is empty"
    exit 1
  fi
  for sf in "${_seed_arr[@]}"; do
    seed_items+=("$sf:$(basename "$sf" .txt)")
  done
elif [[ "$SEED_MODE" == "single" ]]; then
  seed_items+=("$SEED_FILE_SINGLE:$(basename "$SEED_FILE_SINGLE" .txt)")
else
  echo "Unsupported SEED_MODE=$SEED_MODE (expected: multi|single)"
  exit 1
fi

for ZERO_HV in $ZERO_HV_LIST; do
  for NEGATE_MODE in $NEGATE_MODE_LIST; do
    for STRATEGY in $STRATEGY_LIST; do
      for item in "${seed_items[@]}"; do
        SEED_FILE_RUN="${item%%:*}"
        REP_TAG_RUN="${item##*:}"

        JOB_TAG="2d_${STRATEGY}_${NEGATE_MODE}_c${ZERO_HV}_${REP_TAG_RUN}"
        OUT_LOG="${LOG_ROOT}/${JOB_TAG}.log"
        OUTDIR="${OUTPUT_ROOT}/${JOB_TAG}"

        NEGATE_ARGS=()
        if [[ "$NEGATE_MODE" == "both" ]]; then
          read -r -a NEGATE_ARR <<< "$NEGATE_COLS_DEFAULT"
          NEGATE_ARGS=(--negate_cols "${NEGATE_ARR[@]}")
        elif [[ "$NEGATE_MODE" != "none" ]]; then
          echo "Unsupported NEGATE_MODE=$NEGATE_MODE (expected: both|none)"
          exit 1
        fi

        ZERO_ARG=()
        if [[ "$ZERO_HV" == "1" ]]; then
          ZERO_ARG=(--zero-negative-hv)
        elif [[ "$ZERO_HV" == "0" ]]; then
          ZERO_ARG=(--no-zero-negative-hv)
        else
          echo "Unsupported ZERO_HV=$ZERO_HV (expected: 0|1)"
          exit 1
        fi

        UCB_ARGS=()
        if [[ "$UCB_INCLUDE_K0" == "1" ]]; then
          UCB_ARGS=(--ucb_include_k0)
        fi

        FP_ARGS=()
        if [[ -n "$FINGERPRINT_COL" ]]; then
          FP_ARGS=(--fingerprint_col "$FINGERPRINT_COL")
        fi

        WANDB_ARGS=()
        if [[ "$WANDB_ENABLE" == "1" ]]; then
          WANDB_ARGS=(--wandb --wandb_project "$WANDB_PROJECT")
          if [[ -n "$WANDB_ENTITY" ]]; then
            WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
          fi
          if [[ -n "$WANDB_RUN_NAME_PREFIX" ]]; then
            WANDB_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME_PREFIX}_${JOB_TAG}")
          fi
        fi

        EXTRA_ARR=()
        if [[ -n "$EXTRA_ARGS" ]]; then
          read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
        fi

        CMD=(python -u -m src.train
          --data_file "$DATA_FILE"
          --property_cols $PROPERTY_COLS
          "${NEGATE_ARGS[@]}"
          --strategies "$STRATEGY"
          --k_list $K_LIST
          "${ZERO_ARG[@]}"
          --seed_size "$SEED_SIZE"
          --batch_size "$BATCH_SIZE"
          --n_iterations "$N_ITERATIONS"
          --n_replicates "$N_REPLICATES"
          --seed_indices_file "$SEED_FILE_RUN"
          --global_pareto_file "$GLOBAL_PARETO_FILE"
          --device "$DEVICE"
          "${FP_ARGS[@]}"
          "${UCB_ARGS[@]}"
          "${WANDB_ARGS[@]}"
          "${EXTRA_ARR[@]}"
          --output_dir "$OUTDIR"
        )

        CMD_ESCAPED=$(printf '%q ' "${CMD[@]}")
        WRAP_CMD=$(cat <<EOC
source "$CONDA_INIT"
conda activate "$CONDA_ENV"
${CMD_ESCAPED}> $(printf '%q' "$OUT_LOG") 2>&1
EOC
)

        sbatch \
          --job-name="$JOB_TAG" \
          --cpus-per-task="$CPUS_PER_TASK" \
          --time="$TIME_LIMIT" \
          --partition="$PARTITION" \
          --gres="$GRES" \
          -A "$ACCOUNT" \
          --mem="$MEM_PER_JOB" \
          --output="${LOG_ROOT}/${JOB_TAG}_%j.out" \
          --error="${LOG_ROOT}/${JOB_TAG}_%j.err" \
          --wrap "$WRAP_CMD"
      done
    done
  done
done
