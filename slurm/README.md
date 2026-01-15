# Running cohort jobs on Slurm

This folder contains two Slurm helpers:
- `run_cohorts_from_csv.sh` – walks a CSV with `COHORT_ID` values, finds matching cohort folders, and runs `main.py generate_full_body` for each one.
- `compare_cohorts.slurm` – takes a prepared cohort and compares every mouse against wild-type normals (`per_subject` mode).

## Prerequisites
- A Slurm login node with the `mouseSkull_volta` conda env available (the scripts source `~/.bashrc` and `conda activate mouseSkull_volta`).
- A CSV with a `COHORT_ID` column. Each value should appear in the cohort directory names (e.g., `2024/Screen20241223c370_Slc6a4 & wt`).
- Year-based directory layout under your search root (the extractor looks for `<root>/2023`, `<root>/2024`, …).

## 1) Extract full-body volumes for multiple cohorts
Submit the batch job with your CSV (and optionally a custom search root):

```bash
sbatch slurm/run_cohorts_from_csv.sh /path/to/cohorts.csv /path/to/cohort_search_root
```

Notes:
- The search root defaults to `${SLURM_SUBMIT_DIR:-slurm/..}` unless you pass an argument or set `COHORT_SEARCH_ROOT` / `COHORT_BASE_DIR`.
- For each `COHORT_ID`, the script scans year-named subfolders under the search root, de-duplicates matches, and runs:
  ```
  python main.py generate_full_body --cohort-dir <matched_path>
  ```
- Logs land in `logs/run_cohorts_from_csv_<jobid>.out|err`.
- Jobs will exit early if the CSV is missing or the layout does not include year directories.

## (Optional) Clear previous outputs for selected cohorts
If you need to wipe old `full_body_workdir` and `mouse_skull_workdir` folders before re-running:

```bash
bash slurm/clear_cohort_outputs.sh /path/to/cohorts.csv /path/to/cohort_search_root
```

- Add `--dry-run` to see what would be removed without deleting.
- Uses the same `COHORT_ID` lookup logic and year-named layout as `run_cohorts_from_csv.sh`.

## 2) Per-mouse comparison against wild-type normals
After the full-body extraction finishes for your new cohort:

1) Edit `slurm/compare_cohorts.slurm` to point to your data:
   - `REPO` – path to this repository.
   - `REF_DIR` – reference mouse directory.
   - `COHORT_1` – wild-type cohort directory (must contain `males` and `females`).
   - `COHORT_2` – the newly processed cohort (the script will create sex buckets under `COHORT_2/mouse_skull_workdir`).
   - `RESULT_DIR` – where outputs should go.
2) Submit the comparison job:

```bash
sbatch slurm/compare_cohorts.slurm
```

What it does:
- Cleans any `.snakemake` cache, prepares `COHORT_2` images via `generate_full_body`, copies sex folders, then runs:
  ```
  python main.py compare_cohorts --method per_subject --ref_distribution cohort_1 \
    --reference_mouse "$REF_DIR" --cohort_1 "$COHORT_1/<sex>" \
    --cohort_2 "$COHORT_2/mouse_skull_workdir/<sex>" --result_dir "$RESULT_DIR" \
    --fixate_plotting_max 5.0
  ```
- Results and logs are written under `RESULT_DIR` and `logs/compare_cohorts_<jobid>.out|err`.

## Quick checklist
1) Prepare the CSV with `COHORT_ID` values and ensure cohort folders live under year-named directories.
2) Run `sbatch slurm/run_cohorts_from_csv.sh ...` to build full-body volumes.
3) Update paths in `slurm/compare_cohorts.slurm`, then run `sbatch slurm/compare_cohorts.slurm` for per-mouse WT comparisons.
