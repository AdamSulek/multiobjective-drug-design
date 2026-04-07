# Commands

Run from:

```bash
cd archive/pareto-al
```

## 2D

Default matrix:

```bash
bash submit_2d.sh
```

Single seed:

```bash
SEED_MODE=single SEED_FILE_SINGLE=seeds/rep1.txt bash submit_2d.sh
```

Custom project folders:

```bash
PROJECT=table_x_v2 bash submit_2d.sh
```

Limit równoległości (np. max 1 proces naraz):

```bash
MAX_PARALLEL=1 bash submit_2d.sh
```

## 3D

Default matrix:

```bash
bash submit_3d.sh
```

Single seed, one strategy:

```bash
SEED_MODE=single SEED_FILE_SINGLE=seeds/rep0.txt STRATEGY_LIST="ucb" bash submit_3d.sh
```

3D objective/negation control:

```bash
PROPERTY_COLS="score_3GVB score_6D6P score_6GQP" \
NEGATE_MODE_LIST="two all3" \
bash submit_3d.sh
```

Conda env override:

```bash
CONDA_ENV=conda_gpu bash submit_3d.sh
```

Limit równoległości (np. max 2 procesy naraz):

```bash
MAX_PARALLEL=2 bash submit_3d.sh
```

## Reporting

All (tables + plots):

```bash
python -u scripts/report_cases.py --project table_x
```

Only tables:

```bash
python -u scripts/report_cases.py --project table_x --mode tables
```

Only plots:

```bash
python -u scripts/report_cases.py --project table_x --mode plots --normalize-oracle
```

Without `--project` (manual paths):

```bash
python -u scripts/report_cases.py --results-root results --logs-root logs --out-dir results/tables
```

## Direct Unified Run (2D/3D)

Single 2D run (no nohup):

```bash
conda run --no-capture-output -n conda_gpu python -u -m src.train \
  --data_file data/savi_data.parquet \
  --property_cols score_3GVB score_6D6P \
  --negate_cols score_3GVB score_6D6P \
  --strategies ucb \
  --k_list 1 2 3 4 \
  --ucb_include_k0 \
  --seed_size 100 \
  --batch_size 100 \
  --n_iterations 20 \
  --n_replicates 1 \
  --seed_indices_file seeds/rep0.txt \
  --global_pareto_file data/savi_data.parquet \
  --device cuda \
  --fingerprint_col X_ecfp_2 \
  --output_dir results/quick_test/2d_ucb
```

W&B enabled:

```bash
conda run --no-capture-output -n conda_gpu python -u -m src.train \
  --data_file data/savi_data.parquet \
  --property_cols score_3GVB score_6D6P \
  --negate_cols score_3GVB score_6D6P \
  --strategies random ucb ellipse_fast ellipse_directions \
  --wandb \
  --wandb_project multiobjective-drug-design \
  --wandb_run_name quick-2d
```

## Pareto/HV Benchmark

CPU benchmark (2D/3D/4D/5D):

```bash
conda run --no-capture-output -n conda_gpu python -u scripts/benchmark_pareto_hv.py \
  --dims 2 3 4 5 \
  --n_points 1000 3000 5000 \
  --n_candidates 1000 \
  --repeats 5 \
  --out_csv results/benchmark/pareto_hv_benchmark.csv \
  --out_summary_csv results/benchmark/pareto_hv_summary.csv
```

Benchmark + W&B:

```bash
conda run --no-capture-output -n conda_gpu python -u scripts/benchmark_pareto_hv.py \
  --dims 2 3 4 5 \
  --n_points 1000 3000 \
  --repeats 3 \
  --wandb \
  --wandb_project multiobjective-drug-design \
  --wandb_run_name pareto-hv-bench
```
