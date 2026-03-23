#!/bin/bash
set -euo pipefail

# ===== Core parameters =====
SEED_MODE="${SEED_MODE:-multi}"                       # multi | single
SEED_FILE_LIST="${SEED_FILE_LIST:-seeds/rep0.txt seeds/rep1.txt seeds/rep2.txt}"
SEED_FILE_SINGLE="${SEED_FILE_SINGLE:-seeds/rep0.txt}"
PROJECT="${PROJECT:-default}"
RESULTS_BASE="${RESULTS_BASE:-results/$PROJECT}"
LOGS_BASE="${LOGS_BASE:-logs/$PROJECT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_BASE}"
LOG_ROOT="${LOG_ROOT:-$LOGS_BASE}"

K_LIST="${K_LIST:-1 2 3 4}"
N_REPLICATES="${N_REPLICATES:-1}"

# ===== Experiment matrix =====
ZERO_HV_LIST="${ZERO_HV_LIST:-0 1}"
NEGATE_MODE_LIST="${NEGATE_MODE_LIST:-two all3}"     # two | all3 | none
STRATEGY_LIST="${STRATEGY_LIST:-random ucb ellipse_fast ellipse_directions}"

# ===== Runtime/resources =====
MEM_PER_JOB="${MEM_PER_JOB:-260G}"
TIME_MODE="${TIME_MODE:-auto}"                        # auto | fixed
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
TIME_RANDOM="${TIME_RANDOM:-02:00:00}"
TIME_UCB="${TIME_UCB:-12:00:00}"
TIME_ELLIPSE_FAST="${TIME_ELLIPSE_FAST:-06:00:00}"
TIME_ELLIPSE_DIRECTIONS="${TIME_ELLIPSE_DIRECTIONS:-01:00:00}"

CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
PARTITION="${PARTITION:-plgrid-gpu-gh200}"
ACCOUNT="${ACCOUNT:-plgsonata19-gpu-gh200}"
GRES="${GRES:-gpu:1}"

# ===== Environment/data =====
CONDA_INIT="${CONDA_INIT:-/net/storage/pr3/plgrid/plggsanodrugs/miniconda-arm/bin/activate}"
CONDA_ENV="${CONDA_ENV:-savi-arm}"

# PROPERTY_COLS controls selected objectives; NEGATE_MODE chooses which of them are negated.
DATA_FILE="${DATA_FILE:-data/3D/savi_3D_wo_X.parquet}"
X_NPY="${X_NPY:-data/3D/X_uint8.npy}"
PROPERTY_COLS="${PROPERTY_COLS:-score_3GVB score_6D6P score_6GQP}"
NEGATE_COLS_TWO="${NEGATE_COLS_TWO:-score_3GVB score_6D6P}"
NEGATE_COLS_ALL3="${NEGATE_COLS_ALL3:-score_3GVB score_6D6P score_6GQP}"
GLOBAL_PARETO_FILE="${GLOBAL_PARETO_FILE:-data/pareto_global_3D.parquet}"
DEVICE="${DEVICE:-cuda}"

SEED_SIZE="${SEED_SIZE:-100}"
BATCH_SIZE="${BATCH_SIZE:-100}"
N_ITERATIONS="${N_ITERATIONS:-20}"
UCB_INCLUDE_K0="${UCB_INCLUDE_K0:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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
      if [[ "$TIME_MODE" == "fixed" ]]; then
        TIME_LIMIT_RUN="$TIME_LIMIT"
      else
        case "$STRATEGY" in
          random) TIME_LIMIT_RUN="$TIME_RANDOM" ;;
          ucb) TIME_LIMIT_RUN="$TIME_UCB" ;;
          ellipse_fast) TIME_LIMIT_RUN="$TIME_ELLIPSE_FAST" ;;
          ellipse_directions) TIME_LIMIT_RUN="$TIME_ELLIPSE_DIRECTIONS" ;;
          *)
            echo "Unsupported STRATEGY=$STRATEGY"
            exit 1
            ;;
        esac
      fi

      for item in "${seed_items[@]}"; do
        SEED_FILE_RUN="${item%%:*}"
        REP_TAG_RUN="${item##*:}"

        JOB_TAG="3d_${STRATEGY}_${NEGATE_MODE}_c${ZERO_HV}_${REP_TAG_RUN}"
        OUT_LOG="${LOG_ROOT}/${JOB_TAG}.log"
        OUTDIR="${OUTPUT_ROOT}/${JOB_TAG}"

        NEGATE_ARGS=""
        if [[ "$NEGATE_MODE" == "two" ]]; then
          NEGATE_ARGS="--negate_cols ${NEGATE_COLS_TWO}"
        elif [[ "$NEGATE_MODE" == "all3" ]]; then
          NEGATE_ARGS="--negate_cols ${NEGATE_COLS_ALL3}"
        elif [[ "$NEGATE_MODE" != "none" ]]; then
          echo "Unsupported NEGATE_MODE=$NEGATE_MODE (expected: two|all3|none)"
          exit 1
        fi

        if [[ "$ZERO_HV" == "1" ]]; then
          ZERO_ARG="--zero-negative-hv"
        elif [[ "$ZERO_HV" == "0" ]]; then
          ZERO_ARG="--no-zero-negative-hv"
        else
          echo "Unsupported ZERO_HV=$ZERO_HV (expected: 0|1)"
          exit 1
        fi

        UCB_K0_ARG=""
        if [[ "$UCB_INCLUDE_K0" == "1" ]]; then
          UCB_K0_ARG="--ucb_include_k0"
        fi

        WRAP_CMD=$(cat <<EOC
source "$CONDA_INIT"
conda activate "$CONDA_ENV"
python -u -m pal.compare_flexible \
  --data_file "$DATA_FILE" \
  --x_npy "$X_NPY" \
  --property_cols $PROPERTY_COLS \
  $NEGATE_ARGS \
  --strategies "$STRATEGY" \
  --k_list $K_LIST \
  $ZERO_ARG \
  --seed_size "$SEED_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --n_iterations "$N_ITERATIONS" \
  --n_replicates "$N_REPLICATES" \
  --seed_indices_file "$SEED_FILE_RUN" \
  --global_pareto_file "$GLOBAL_PARETO_FILE" \
  --device "$DEVICE" \
  $UCB_K0_ARG \
  $EXTRA_ARGS \
  --output_dir "$OUTDIR" \
  > "$OUT_LOG" 2>&1
EOC
)

        sbatch \
          --job-name="$JOB_TAG" \
          --cpus-per-task="$CPUS_PER_TASK" \
          --time="$TIME_LIMIT_RUN" \
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
