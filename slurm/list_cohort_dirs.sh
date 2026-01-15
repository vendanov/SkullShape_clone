#!/usr/bin/env bash
# List cohort directories matching COHORT_IDs from a CSV.
# Re-exec with bash if invoked via sh/dash.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail
shopt -s nullglob

usage() {
  cat >&2 <<'EOF'
Usage: list_cohort_dirs.sh cohorts.csv [cohort_search_root]

- cohorts.csv: CSV file containing a COHORT_ID column.
- cohort_search_root (optional): base directory containing year-named subfolders (e.g., 2023, 2024) with cohort directories.
  Defaults to ${COHORT_SEARCH_ROOT:-${COHORT_BASE_DIR:-$REPO}} where REPO is the repo root (slurm/..).
EOF
  exit 1
}

if (( $# < 1 || $# > 2 )); then
  usage
fi

csv_path=$1
if [[ ! -f "$csv_path" ]]; then
  echo "CSV file not found: $csv_path" >&2
  exit 1
fi

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
mapfile -t year_dirs < <(find "$cohort_search_root" -maxdepth 1 -mindepth 1 -type d -regextype posix-extended -regex '.*/[0-9]{4}$')
if (( ${#year_dirs[@]} == 0 )); then
  echo "No year-named directories (e.g., 2023) found under $cohort_search_root" >&2
  exit 1
fi

declare -A seen_paths=()
any_found=0

for cohort_id in "${cohort_ids[@]}"; do
  found_for_id=0
  for year_dir in "${year_dirs[@]}"; do
    for cohort_dir in "$year_dir"/*"$cohort_id"*; do
      [[ -d "$cohort_dir" ]] || continue
      abs="$(cd "$cohort_dir" && pwd)"
      if [[ -n ${seen_paths[$abs]:-} ]]; then
        continue
      fi
      seen_paths["$abs"]=1
      any_found=1
      found_for_id=1
      echo "$abs"
    done
  done
  if (( found_for_id == 0 )); then
    echo "Warning: no cohort directory found for cohort ID '$cohort_id'" >&2
  fi
done

if (( any_found == 0 )); then
  echo "No cohort directories matched any IDs in $csv_path under $cohort_search_root" >&2
  exit 1
fi
