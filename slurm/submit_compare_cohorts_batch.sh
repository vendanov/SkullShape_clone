#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <cohorts_root_dir> [slurm_script]"
  exit 1
fi

COHORTS_ROOT="$1"
SLURM_SCRIPT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compare_cohorts.slurm}"

if [[ ! -d "$COHORTS_ROOT" ]]; then
  echo "Cohorts root not found: $COHORTS_ROOT"
  exit 1
fi

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Slurm script not found: $SLURM_SCRIPT"
  exit 1
fi

declare -A jobs

for cohort_dir in "$COHORTS_ROOT"/*/; do
  [[ -d "$cohort_dir" ]] || continue
  cohort_name="$(basename "$cohort_dir")"
  output_dir="$cohort_dir/_mahalanobis_results"
  mkdir -p "$output_dir"
  job_id="$(sbatch --parsable "$SLURM_SCRIPT" "$cohort_dir" "$output_dir")"
  jobs["$job_id"]="$cohort_name"
  echo "[INFO] Submitted $cohort_name as job $job_id"
done

if [[ ${#jobs[@]} -eq 0 ]]; then
  echo "[WARN] No cohorts found under $COHORTS_ROOT"
  exit 0
fi

while [[ ${#jobs[@]} -gt 0 ]]; do
  for job_id in "${!jobs[@]}"; do
    if ! squeue -h -j "$job_id" >/dev/null 2>&1; then
      echo "[INFO] Cohort ${jobs[$job_id]} finished (job $job_id)"
      unset jobs["$job_id"]
    fi
  done
  sleep 60
done

echo "[INFO] All cohorts finished."
