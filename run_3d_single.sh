#!/bin/bash -l
#SBATCH --job-name=pal3d
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH -A plgsonata19-gpu-gh200
#SBATCH --mem=260G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONDA_INIT="${CONDA_INIT:-/net/storage/pr3/plgrid/plggsanodrugs/miniconda-arm/bin/activate}"
CONDA_ENV="${CONDA_ENV:-savi-arm}"
source "$CONDA_INIT"
conda activate "$CONDA_ENV"

STRATEGY="${STRATEGY:?}"               # random | ucb | ellipse_fast | ellipse_directions
ZERO_HV="${ZERO_HV:?}"                 # 0 | 1
NEGATE_MODE="${NEGATE_MODE:?}"         # two | all3 | none

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
N_REPLICATES="${N_REPLICATES:-3}"
K_LIST="${K_LIST:-1 2 3 4}"
SEED_FILES="${SEED_FILES:-seeds/rep0.txt seeds/rep1.txt seeds/rep2.txt}"
UCB_INCLUDE_K0="${UCB_INCLUDE_K0:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
LOG_ROOT="${LOG_ROOT:-logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

read -r -a PROPERTY_ARR <<< "$PROPERTY_COLS"
if [[ "${#PROPERTY_ARR[@]}" -ne 3 ]]; then
  echo "For 3D run, PROPERTY_COLS must contain exactly 3 columns. Got: $PROPERTY_COLS"
  exit 1
fi

NEGATE_ARGS=()
if [[ "$NEGATE_MODE" == "two" ]]; then
  read -r -a NEGATE_ARR <<< "$NEGATE_COLS_TWO"
  NEGATE_ARGS=(--negate_cols "${NEGATE_ARR[@]}")
elif [[ "$NEGATE_MODE" == "all3" ]]; then
  read -r -a NEGATE_ARR <<< "$NEGATE_COLS_ALL3"
  NEGATE_ARGS=(--negate_cols "${NEGATE_ARR[@]}")
elif [[ "$NEGATE_MODE" == "none" ]]; then
  NEGATE_ARGS=()
else
  echo "Unsupported NEGATE_MODE=$NEGATE_MODE (expected: two|all3|none)"
  exit 1
fi

if [[ "$ZERO_HV" == "1" ]]; then
  ZERO_ARG=(--zero-negative-hv)
  ZERO_TAG="clip1"
elif [[ "$ZERO_HV" == "0" ]]; then
  ZERO_ARG=(--no-zero-negative-hv)
  ZERO_TAG="clip0"
else
  echo "Unsupported ZERO_HV=$ZERO_HV (expected: 0|1)"
  exit 1
fi

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT"

OUTDIR="$OUTPUT_ROOT/3d_${STRATEGY}_${NEGATE_MODE}_${ZERO_TAG}"
LOGFILE="$LOG_ROOT/3d_${STRATEGY}_${NEGATE_MODE}_${ZERO_TAG}.log"

read -r -a K_ARR <<< "$K_LIST"
read -r -a SEED_ARR <<< "$SEED_FILES"

CMD=(python -u -m pal.compare_flexible
  --data_file "$DATA_FILE"
  --x_npy "$X_NPY"
  --property_cols "${PROPERTY_ARR[@]}"
  "${NEGATE_ARGS[@]}"
  --strategies "$STRATEGY"
  --k_list "${K_ARR[@]}"
  "${ZERO_ARG[@]}"
  --seed_size "$SEED_SIZE"
  --batch_size "$BATCH_SIZE"
  --n_iterations "$N_ITERATIONS"
  --n_replicates "$N_REPLICATES"
  --seed_indices_files "${SEED_ARR[@]}"
  --global_pareto_file "$GLOBAL_PARETO_FILE"
  --device "$DEVICE"
  --output_dir "$OUTDIR"
)

if [[ "$UCB_INCLUDE_K0" == "1" ]]; then
  CMD+=(--ucb_include_k0)
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
  CMD+=("${EXTRA_ARR[@]}")
fi

"${CMD[@]}" > "$LOGFILE" 2>&1
