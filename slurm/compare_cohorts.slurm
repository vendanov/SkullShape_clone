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
REPO_FULL_BODY="/home/varhai/mouse_full_body"
REF_DIR="/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/reference_image"
COHORT_1="/mnt/magnus/ccp/Igor/male_female_new_alg"
COHORT_2="/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/2024/Screen20241223c370_Slc6a4 & wt"
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

if [[ -d ".snakemake" ]]; then
  rm -rf ".snakemake"
fi
# Now prepare cohort_2 by making images.

echo "[INFO] Preparing cohort_2 images under: $COHORT_2"

python "$REPO/main.py" generate_full_body \
  --cohort-dir "$COHORT_2" 




echo "[INFO] Preparing cohort_1 sex buckets under: $COHORT_2"

bash "$REPO/copy_sex_folders.sh" "$COHORT_2/mouse_skull_workdir"

# # now cohort_2 should 

echo "[INFO] Running compare_cohorts (per_subject)"
python "$REPO/main.py" compare_cohorts \
  --reference_mouse "$REF_DIR" \
  --cohort_1 "$COHORT_1/females" \
  --cohort_2 "$COHORT_2/mouse_skull_workdir/females" \
  --result_dir "$RESULT_DIR" \
  --method per_subject \
  --ref_distribution cohort_1\
  --fixate_plotting_max 5.0

echo "[INFO] Running compare_cohorts (per_subject)"
python "$REPO/main.py" compare_cohorts \
  --reference_mouse "$REF_DIR" \
  --cohort_1 "$COHORT_1/males" \
  --cohort_2 "$COHORT_2/mouse_skull_workdir/males" \
  --result_dir "$RESULT_DIR" \
  --method per_subject \
  --ref_distribution cohort_1\
  --fixate_plotting_max 5.0

