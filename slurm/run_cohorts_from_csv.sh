#!/usr/bin/env bash
#SBATCH --job-name=run_cohorts_from_csv
#SBATCH --cpus-per-task=24
#SBATCH --mem=725000M
#SBATCH --time=240:00:00
#SBATCH --nice=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=varhai@img.cas.cz

# Run generate_full_body for cohort directories listed in a CSV's COHORT_ID column.
set -euo pipefail
shopt -s nullglob

mkdir -p logs

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 path/to/cohorts.csv [cohort_search_root]" >&2
  exit 1
fi

csv_path=$1
if [[ ! -f "$csv_path" ]]; then
  echo "CSV file not found: $csv_path" >&2
  exit 1
fi

# Environment
source /home/varhai/.bashrc
conda activate mouseSkull_volta


script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_default="$(cd "${SLURM_SUBMIT_DIR:-$script_dir/..}" && pwd)"
REPO="${REPO:-$repo_default}"

cohort_search_root="${2:-${COHORT_SEARCH_ROOT:-${COHORT_BASE_DIR:-$REPO}}}"
if [[ ! -d "$cohort_search_root" ]]; then
  echo "Cohort search directory not found: $cohort_search_root" >&2
  exit 1
fi

cohort_search_root="$(cd "$cohort_search_root" && pwd)"

# Collect unique cohort IDs from the CSV.
mapfile -t cohort_ids < <(
  python - <<'PY' "$csv_path"
import csv
import sys

csv_path = sys.argv[1]

with open(csv_path, newline='') as fh:
    reader = csv.DictReader(fh)
    if reader.fieldnames is None or 'COHORT_ID' not in reader.fieldnames:
        sys.stderr.write("COHORT_ID column not found in CSV\n")
        sys.exit(1)
    seen = {}
    for row in reader:
        cid = (row.get('COHORT_ID') or '').strip()
        if cid:
            seen.setdefault(cid, None)

for cid in seen:
    print(cid)
PY
)

if (( ${#cohort_ids[@]} == 0 )); then
  echo "No cohort IDs found in $csv_path" >&2
  exit 1
fi

# Discover year-named subdirectories under the cohort search root.
year_dirs=()
while IFS= read -r entry; do
  year_dirs+=("$entry")
done < <(find "$cohort_search_root" -maxdepth 1 -mindepth 1 -type d -regextype posix-extended -regex '.*/[0-9]{4}$')

if (( ${#year_dirs[@]} == 0 )); then
  echo "No year-named directories (e.g., 2023) found under $cohort_search_root" >&2
  exit 1
fi

declare -A processed_dirs=()

for cohort_id in "${cohort_ids[@]}"; do
  found_any=0

  for year_dir in "${year_dirs[@]}"; do
    for cohort_dir in "$year_dir"/*"$cohort_id"*; do
      [[ -d "$cohort_dir" ]] || continue

      # Resolve to an absolute path for deduplication.
      abs_cohort_dir="$(cd "$cohort_dir" && pwd)"
      if [[ -n ${processed_dirs[$abs_cohort_dir]:-} ]]; then
        continue
      fi

      processed_dirs["$abs_cohort_dir"]=1
      found_any=1

      echo "Running generate_full_body for cohort ID '$cohort_id' at $abs_cohort_dir"
      python "$REPO/main.py" generate_full_body --cohort-dir "$abs_cohort_dir"
    done
  done

  if (( found_any == 0 )); then
    echo "Warning: no cohort directory found for cohort ID '$cohort_id'" >&2
  fi
done
