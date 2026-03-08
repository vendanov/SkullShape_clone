# AGENTS.md

This file guides coding agents working in `mouseSkullPipelines`.

## Project Snapshot

- Primary entry point: `main.py`
- Core workflows: `snakemake_workflows/generic_coregistration.smk`, `snakemake_workflows/mouse_full_body.smk`, `snakemake_workflows/extract_head_from_full_body.smk`
- Shared config: `smk_configs/mouse_skull.yaml`
- Algorithm code: `mouse_skull_pipelines/scripts/`
- Mesh transform helper: `support/convert_mesh.py`

## Target Project Structure

- Current issue:
  - `main.py` contains too much orchestration and domain logic.
- Preferred structure for ongoing refactors:
  - `main.py` or `mouse_skull_pipelines/cli.py`: thin CLI entrypoint only
  - `mouse_skull_pipelines/commands/`: command handlers (`compare_cohorts.py`, `one_cohort_statistics.py`, `generate_full_body.py`, `visualize_subject.py`)
  - `mouse_skull_pipelines/services/`: workflow orchestration (Snakemake invocations, config assembly, retries, job control)
  - `mouse_skull_pipelines/analysis/`: statistics and cohort analysis logic (KS/t-test/Mahalanobis/aggregations)
  - `mouse_skull_pipelines/io/`: path discovery, file validation, result writing
  - `mouse_skull_pipelines/visualization/`: GIF and plotting code
  - `mouse_skull_pipelines/config/`: config schema/defaults/loaders
  - keep low-level mesh/image operations in existing `mouse_skull_pipelines/scripts/` unless explicitly migrated
- Tests (when added/expanded):
  - `tests/commands/`, `tests/services/`, `tests/analysis/`

## Environment and Dependencies

- Python: 3.9+
- Install Python deps:
  - `pip install -r requirements.txt`
- External tools expected by workflows:
  - `snakemake`
  - 3D Slicer (configured via `slicer_core`)
  - Registration tools referenced by workflow/config (NiftyReg/Elastix/ANTs depending on run path)
- Runtime assumption:
  - Snakemake pipelines are primarily designed for Linux/HPC environments.

## Data and Path Conventions

- Cohort layout:
  - `<cohort_dir>/<subject>/ct_image.nii.gz`
- Reference layout:
  - `oriented_image.nii.gz`
  - `bone_coregistration_mask.nii.gz`
  - `oriented_mesh.obj`
  - `ref_landmarks.pts` (optional for some `main.py` paths)

## Common Commands

- Compare two cohorts:
  - `python main.py compare_cohorts --cohort_1 <path> --cohort_2 <path> --reference_mouse <ref_dir> --result_dir <out_dir>`
- One cohort variability:
  - `python main.py one_cohort_statistics --cohort <path> --reference_mouse <ref_dir> --result_dir <out_dir>`
- Full-body preprocessing:
  - `python main.py generate_full_body --cohort <path> --reference_mouse <ref_dir> --result_dir <out_dir>`
- Direct Snakemake run:
  - `snakemake --snakefile snakemake_workflows/generic_coregistration.smk --configfile smk_configs/mouse_skull.yaml --cores 8 --keep-going`

## CLI Design Standard

- For new CLI surfaces, prefer `Typer` (with type hints) over raw `argparse`.
- Keep the existing `python main.py <command>` UX stable during migration.
- Add new commands as typed Typer subcommands and keep argument names consistent with current flags where practical.
- Preserve non-interactive batch/HPC behavior:
  - no required prompts
  - deterministic defaults
  - clear exit codes on failure
- If `argparse` code is touched, only do minimal compatibility edits unless explicitly asked for a full CLI migration.

## Agent Working Rules

- Prefer minimal, targeted edits. Do not refactor unrelated code.
- Preserve CLI behavior and existing output file names unless explicitly requested.
- When changing workflow paths or file names, verify `main.py` and Snakemake rules stay consistent.
- Avoid hardcoding machine-specific absolute paths in committed files.
- Keep compatibility with Linux execution for Snakemake/SLURM usage.
- For `main.py` changes:
  - do not add new heavy business logic directly into `main.py`
  - extract reusable logic into modules from the target structure above
  - keep `main.py` focused on argument parsing, command dispatch, and top-level error handling

## Validation Checklist After Changes

- If Python code changed:
  - run at least a syntax check for touched files, e.g. `python -m py_compile <file.py>`
- If Snakemake code changed:
  - dry-run the relevant workflow when possible, e.g. `snakemake -n --snakefile <workflow.smk> --configfile <config.yaml>`
- If CLI args changed:
  - verify CLI `--help` output and call sites in `main.py`
- If code was moved out of `main.py`:
  - verify imports have no circular dependency
  - run at least one representative command end-to-end (or dry-run equivalent) after refactor

## Scope Notes

- `notebooks/`, `test_notebooks/`, and `matlab/` may contain exploratory or legacy artifacts.
- Production-critical path is centered around `main.py` + `snakemake_workflows/` + `mouse_skull_pipelines/scripts/`.
