#!/usr/bin/env bash
# One-time helper to copy immediate subfolders containing "_M_" or "_F_"
# into males/ and females/ directories under each provided root.
#
# Usage:
#   ./copy_sex_folders.sh [-o OUT_PARENT] /path/to/cohort1 [/path/to/cohort2 ...]
#
# Notes:
# - Uses Python shutil.copytree to avoid shell quoting pitfalls on paths with spaces or '&'.
# - Change the helper below if you prefer moving instead of copying.

set -euo pipefail

OUT_PARENT=""

usage() {
  echo "Usage: $0 [-o OUT_PARENT] ROOT_DIR [ROOT_DIR ...]" >&2
}

# Copy directory tree using Python to avoid quoting issues with special characters.
copy_subject_tree() {
  local src="$1"
  local dst="$2"
  python3 - "$src" "$dst" <<'PY'
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
if dst.exists():
    sys.exit(0)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(src, dst, copy_function=shutil.copy2, symlinks=True)
PY
}

# Parse options
while getopts ":o:h" opt; do
  case $opt in
    o) OUT_PARENT="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done
shift $((OPTIND-1))

if (( "$#" < 1 )); then
  usage
  exit 1
fi

for root in "$@"; do
  if [[ ! -d "$root" ]]; then
    echo "Skip: $root is not a directory" >&2
    continue
  fi

  if [[ -n "$OUT_PARENT" ]]; then
    dest_base="$OUT_PARENT"
  else
    dest_base="$root"
  fi

  mkdir -p "$dest_base/males" "$dest_base/females"

  # Process immediate children only, excluding target folders themselves
  while IFS= read -r -d '' d; do
    name=$(basename "$d")
    case "$name" in
      *_M_*)
        dest="$dest_base/males/$name"
        if [[ -e "$dest" ]]; then
          echo "Exists, skipping: $dest"
        else
          echo "Copy: $name -> $dest_base/males/"
          copy_subject_tree "$d" "$dest"
        fi
        ;;
      *_F_*)
        dest="$dest_base/females/$name"
        if [[ -e "$dest" ]]; then
          echo "Exists, skipping: $dest"
        else
          echo "Copy: $name -> $dest_base/females/"
          copy_subject_tree "$d" "$dest"
        fi
        ;;
      *) ;;
    esac
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d ! -name males ! -name females -print0)
done
