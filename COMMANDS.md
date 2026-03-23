# Commands

Run from:

```bash
cd archive/pareto-al
```

## 2D

Default matrix:

```bash
sbatch submit_2d.sh
```

Single seed:

```bash
SEED_MODE=single SEED_FILE_SINGLE=seeds/rep1.txt sbatch submit_2d.sh
```

Custom project folders:

```bash
PROJECT=table_x_v2 sbatch submit_2d.sh
```

## 3D

Default matrix:

```bash
sbatch submit_3d.sh
```

Single seed, one strategy:

```bash
SEED_MODE=single SEED_FILE_SINGLE=seeds/rep0.txt STRATEGY_LIST="ucb" sbatch submit_3d.sh
```

3D objective/negation control:

```bash
PROPERTY_COLS="score_3GVB score_6D6P score_6GQP" \
NEGATE_MODE_LIST="two all3" \
sbatch submit_3d.sh
```

Time overrides (3D):

```bash
TIME_MODE=fixed TIME_LIMIT=08:00:00 sbatch submit_3d.sh
```

```bash
TIME_UCB=24:00:00 TIME_ELLIPSE_FAST=08:00:00 sbatch submit_3d.sh
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
