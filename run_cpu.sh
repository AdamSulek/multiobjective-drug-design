#!/bin/bash -l
#SBATCH --job-name=run_cpu
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --partition=plgrid
#SBATCH -A plgsonata19-cpu
#SBATCH --mem=160G
#SBATCH --output=logs/cpu_table_x_2d.out
#SBATCH --error=logs/cpu_table_x_2d.err


source /net/storage/pr3/plgrid/plggsanodrugs/miniconda/etc/profile.d/conda.sh
conda activate savi

python -u -m pal.compare_flexible \
  --data_file data/savi_data.parquet \
  --property_cols score_3GVB score_6D6P \
  --negate_cols score_3GVB score_6D6P \
  --strategies random ucb ellipse_fast ellipse_directions \
  --k_list 1 2 3 4 \
  --ucb_include_k0 \
  --seed_size 100 \
  --batch_size 100 \
  --n_iterations 20 \
  --n_replicates 3 \
  --seed_indices_files seeds/rep0.txt seeds/rep1.txt seeds/rep2.txt \
  --global_pareto_file data/savi_data.parquet \
  --device cpu \
  --fingerprint_col X_ecfp_2 \
  --output_dir results/cpu_table_x_2d \
  > logs/cpu_table_x_2d.log
