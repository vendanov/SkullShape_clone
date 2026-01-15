#!/usr/bin/env bash
# Clear previous full-body and head extraction outputs for cohorts listed in a CSV.
# Re-exec with bash if invoked with sh/dash.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail
shopt -s nullglob

usage() {
  cat >&2 <<'EOF'
Usage: clear_cohort_outputs.sh cohorts.csv [cohort_search_root] [--dry-run]

- cohorts.csv: CSV file containing a COHORT_ID column.
- cohort_search_root (optional): base directory that contains year-named subfolders with cohort directories.
  Defaults to ${COHORT_SEARCH_ROOT:-${COHORT_BASE_DIR:-$REPO}} where REPO is the repo root (slurm/..).
- --dry-run: report what would be removed without deleting anything.
EOF
  exit 1
}

dry_run=0
positional=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    --help|-h) usage ;;
    *) positional+=("$arg") ;;
  esac
done

if (( ${#positional[@]} < 1 || ${#positional[@]} > 2 )); then
  usage
fi

csv_path=${positional[0]}
if [[ ! -f "$csv_path" ]]; then
  echo "CSV file not found: $csv_path" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_default="$(cd "${SLURM_SUBMIT_DIR:-$script_dir/..}" && pwd)"
REPO="${REPO:-$repo_default}"

cohort_search_root="${positional[1]:-${COHORT_SEARCH_ROOT:-${COHORT_BASE_DIR:-$REPO}}}"
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

declare -A processed_dirs=()

for cohort_id in "${cohort_ids[@]}"; do
  found_any=0
  for year_dir in "${year_dirs[@]}"; do
    for cohort_dir in "$year_dir"/*"$cohort_id"*; do
      [[ -d "$cohort_dir" ]] || continue

      abs_cohort_dir="$(cd "$cohort_dir" && pwd)"
      if [[ -n ${processed_dirs[$abs_cohort_dir]:-} ]]; then
        continue
      fi
      processed_dirs["$abs_cohort_dir"]=1
      found_any=1

      targets=(
        "$abs_cohort_dir/full_body_workdir"
        "$abs_cohort_dir/mouse_skull_workdir"
      )

      for path in "${targets[@]}"; do
        if [[ -d "$path" ]]; then
          if (( dry_run )); then
            echo "[DRY-RUN] Would remove: $path"
          else
            echo "[INFO] Removing: $path"
            rm -rf -- "$path"
          fi
        fi
      done
    done
  done

  if (( found_any == 0 )); then
    echo "Warning: no cohort directory found for cohort ID '$cohort_id'" >&2
  fi
done

if (( dry_run )); then
  echo "Dry-run complete; no files were deleted."
fi
