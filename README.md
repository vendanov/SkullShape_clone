# mouseSkullPipelines

Pipeline for mouse skull registration, cohort comparison, full-body preprocessing, and visualization.

Primary entry points:
- `main.py`
- `mouse_skull_pipelines/cli.py`

Core workflows:
- `snakemake_workflows/generic_coregistration.smk`
- `snakemake_workflows/mouse_full_body.smk`
- `snakemake_workflows/extract_head_from_full_body.smk`

Common commands:

```bash
python main.py compare_cohorts --cohort_1 <path> --cohort_2 <path> --reference_mouse <ref_dir> --result_dir <out_dir>
python main.py one_cohort_statistics --cohort <path> --reference_mouse <ref_dir> --result_dir <out_dir>
python main.py generate_full_body --cohort-dir <path>
```

Variability GIFs from `one_cohort_statistics` use the cohort face-centroid standard deviation. Use `--upper_limit` to fix the colorbar maximum across runs, and `--std_cmap` to choose a Matplotlib colormap for the variability map:

```bash
python main.py one_cohort_statistics \
  --cohort <path> \
  --reference_mouse <ref_dir> \
  --result_dir <out_dir> \
  --upper_limit 0.4 \
  --std_cmap magma
```

Environment:
- Python 3.9+
- `pip install -r requirements.txt`
- external tools required by the workflows: `snakemake`, 3D Slicer, and the configured registration backends

## TODO

- Keep `python main.py <command>` stable while continuing the CLI migration toward `mouse_skull_pipelines/cli.py`.
- Reduce `main.py` to a thin entrypoint and move orchestration and analysis code into `commands/`, `services/`, `analysis/`, and `io/`.
- Add or expand tests under `tests/commands/`, `tests/services/`, and `tests/analysis/` as logic is moved out of `main.py`.
- Keep Linux/HPC compatibility for Snakemake and SLURM execution paths.
- Preserve output filenames and workflow path conventions unless there is an explicit migration plan.
- Keep the batch submission helper from launching SLURM for cohorts that already have Mahalanobis CSV outputs when `--skip-existing-mahalanobis` is used in [slurm/submit_compare_cohorts_batch.sh](C:/Users/varhai/mouseSkullPipelines/slurm/submit_compare_cohorts_batch.sh).
