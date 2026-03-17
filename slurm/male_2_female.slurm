#!/bin/bash
#SBATCH --job-name=male_female
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=240:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=varhai@img.cas.cz

set -euo pipefail

# Paths (edit as needed or override via args)
# Usage: sbatch compare_cohorts.slurm <cohort_path> <output_path>

# cpus set to 12 for 4 tasks
REPO="/home/varhai/mouse_skull_pipelines"
REPO_FULL_BODY="/home/varhai/mouse_full_body"
REF_DIR="/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/reference_image"
COHORT_1="/mnt/magnus/ccp/Igor/male_female_new_alg"
COHORT_2="${1:-/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/2024/Screen20241111c364_Col2a1 & Sp7 & wt}"
RESULT_DIR="/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/male_female03_2026results"

mkdir -p "$RESULT_DIR"


# Ensure log directory exists relative to submission dir
mkdir -p logs





# Environment
source /home/varhai/.bashrc
conda activate mouseSkull_volta

# Optional: clean snakemake artifacts if running from repo
if [[ -d "$REPO/.snakemake" ]]; then
  rm -rf "$REPO/.snakemake"
fi

if [[ -d ".snakemake" ]]; then
  rm -rf ".snakemake"
fi
# Now prepare cohort_2 by making images.


# # now cohort_2 should 


python "$REPO/main.py" compare_cohorts \
  --reference_mouse "$REF_DIR" \
  --cohort_1 "$COHORT_1/females" \
  --cohort_2 "$COHORT_1/males" \
  --result_dir "$RESULT_DIR" 


echo "[INFO] Running one_coh (fem)"
python "$REPO/main.py" one_cohort_statistics \
  --reference_mouse "$REF_DIR" \
  --cohort "$COHORT_1/females" \
  --result_dir "$RESULT_DIR/female_res" \
  --mahalanobis_summary_csv "$RESULT_DIR/female_res/stat.csv" \
  --upper_limit 0.5 \
  --fast_outliers \
  --skip_variability_gif \
  --save_statics 

echo "[INFO] Running one_coh (m)"

python "$REPO/main.py" one_cohort_statistics \
  --reference_mouse "$REF_DIR" \
  --cohort "$COHORT_1/males" \
  --result_dir "$RESULT_DIR/male_res" \
  --mahalanobis_summary_csv "$RESULT_DIR/male_res/stat.csv" \
  --upper_limit 0.5 \
  --skip_variability_gif \
  --fast_outliers \
  --save_statics