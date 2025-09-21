#!/bin/bash
#SBATCH --job-name=compare_cohorts
#SBATCH --cpus-per-task=24
#SBATCH --mem=725000M
#SBATCH --time=240:00:00
#SBATCH --nice=1000
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=varhai@img.cas.cz

set -euo pipefail

# Paths (edit as needed)
REPO="/home/varhai/mouse_skull_pipelines"
REF_DIR="/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/reference_image"
COHORT_1="/mnt/magnus/ccp/Igor/wt_normals"
COHORT_2="/mnt/magnus/ccp/Igor/michaela_genes_wd/Nipal2"
RESULT_DIR="/home/varhai/results_tmp"

# Ensure log directory exists relative to submission dir
mkdir -p logs





# Environment
source /home/varhai/.bashrc
conda activate mouseSkull_volta

# Optional: clean snakemake artifacts if running from repo
if [[ -d "$REPO/.snakemake" ]]; then
  rm -rf "$REPO/.snakemake"
fi

echo "[INFO] Preparing cohort_1 sex buckets under: $COHORT_1"
"$REPO/copy_sex_folders.sh" "$COHORT_1"

echo "[INFO] Running compare_cohorts (per_subject)"
python "$REPO/main.py" compare_cohorts \
  --reference_mouse "$REF_DIR" \
  --cohort_1 "$COHORT_1" \
  --cohort_2 "$COHORT_2" \
  --result_dir "$RESULT_DIR" \
  --method per_subject \
  --ref_distribution cohort_1 \
  --fixate_plotting_max 5.0

