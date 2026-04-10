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
# Runtime hints retained for compatibility with old command examples.
TIME_MODE="${TIME_MODE:-auto}"                        # ignored by nohup mode
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"                 # ignored by nohup mode
TIME_RANDOM="${TIME_RANDOM:-02:00:00}"               # ignored by nohup mode
TIME_UCB="${TIME_UCB:-12:00:00}"                     # ignored by nohup mode
TIME_ELLIPSE_FAST="${TIME_ELLIPSE_FAST:-06:00:00}"   # ignored by nohup mode
TIME_ELLIPSE_DIRECTIONS="${TIME_ELLIPSE_DIRECTIONS:-01:00:00}"  # ignored by nohup mode

# ===== Environment/data =====
CONDA_ENV="${CONDA_ENV:-conda_gpu}"

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
MAX_PARALLEL="${MAX_PARALLEL:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-2}"
WANDB_ENABLE="${WANDB_ENABLE:-0}"                    # 0 | 1
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME_PREFIX="${WANDB_RUN_NAME_PREFIX:-}"

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found in PATH"
  exit 1
fi

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer. Got: $MAX_PARALLEL"
  exit 1
fi

active_pids=()

reap_finished_pids() {
  local alive=()
  local pid
  for pid in "${active_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  active_pids=("${alive[@]}")
}

wait_for_slot() {
  while true; do
    reap_finished_pids
    if (( ${#active_pids[@]} < MAX_PARALLEL )); then
      break
    fi
    sleep "$WAIT_SECONDS"
  done
}

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

        CMD=(conda run --no-capture-output -n "$CONDA_ENV" python -u -m src.train
          --data_file "$DATA_FILE"
          --x_npy "$X_NPY"
          --property_cols $PROPERTY_COLS
          --strategies "$STRATEGY"
          --k_list $K_LIST
          $ZERO_ARG
          --seed_size "$SEED_SIZE"
          --batch_size "$BATCH_SIZE"
          --n_iterations "$N_ITERATIONS"
          --n_replicates "$N_REPLICATES"
          --seed_indices_file "$SEED_FILE_RUN"
          --global_pareto_file "$GLOBAL_PARETO_FILE"
          --device "$DEVICE"
          --output_dir "$OUTDIR"
        )

        if [[ -n "$NEGATE_ARGS" ]]; then
          read -r -a NEGATE_ARR <<< "$NEGATE_ARGS"
          CMD+=("${NEGATE_ARR[@]}")
        fi

        if [[ -n "$UCB_K0_ARG" ]]; then
          CMD+=(--ucb_include_k0)
        fi

        if [[ -n "$EXTRA_ARGS" ]]; then
          read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
          CMD+=("${EXTRA_ARR[@]}")
        fi

        if [[ "$WANDB_ENABLE" == "1" ]]; then
          CMD+=(--wandb)
          if [[ -n "$WANDB_PROJECT" ]]; then
            CMD+=(--wandb_project "$WANDB_PROJECT")
          fi
          if [[ -n "$WANDB_ENTITY" ]]; then
            CMD+=(--wandb_entity "$WANDB_ENTITY")
          fi
          if [[ -n "$WANDB_RUN_NAME_PREFIX" ]]; then
            CMD+=(--wandb_run_name "${WANDB_RUN_NAME_PREFIX}_${JOB_TAG}")
          else
            CMD+=(--wandb_run_name "$JOB_TAG")
          fi
        fi

        wait_for_slot
        nohup "${CMD[@]}" > "$OUT_LOG" 2>&1 &
        PID=$!
        active_pids+=("$PID")
        echo "$PID" > "${LOG_ROOT}/${JOB_TAG}.pid"
        echo "Started $JOB_TAG pid=$PID log=$OUT_LOG"
      done
    done
  done
done

wait
