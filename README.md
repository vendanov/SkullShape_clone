# mouseSkullPipelines

`main.py` is the project CLI entry point for registration, cohort comparison, and visualization.

It orchestrates:
- Snakemake workflows (`generic_coregistration.smk`, `mouse_full_body.smk`, `extract_head_from_full_body.smk`)
- mesh processing and generalized Procrustes alignment
- cohort statistics and GIF exports

## What `main.py` Runs

Commands:
- `compare_cohorts`
- `one_cohort_statistics`
- `visualize_subject`
- `generate_full_body`

Direct workflow linkage:
- `compare_cohorts` -> runs registration on both cohorts, then computes group or per-subject stats
- `one_cohort_statistics` -> runs registration on one cohort, then computes variability map
- `visualize_subject` -> no Snakemake run; reads already-registered meshes
- `generate_full_body` -> runs full-body conversion/detection and head extraction workflows

### What Each Command Computes

`compare_cohorts` has two modes:

- `--method group`:
  - Runs `generic_coregistration.smk` for both cohorts to produce `head_affine_registered.obj`.
  - Builds cohort means:
    - per-cohort mean mesh (`cohort_1_mean.obj`, `cohort_2_mean.obj`) after Procrustes alignment.
    - then rigidly aligns all subjects from both cohorts together before statistical testing.
  - Computes per-face statistics (face = triangle cell):
    - Face centroid per subject: mean of that face's 3D vertex coordinates.
    - Mean shift vector per face:
      - `delta_face = mean(centroid_face in cohort_1) - mean(centroid_face in cohort_2)` (x, y, z).
      - `mean_diff_<cohort1>_<cohort2>.gif` visualizes `||delta_face||`.
    - Kolmogorov-Smirnov test per face:
      - Two-sample KS on each axis independently (`x`, `y`, `z`) across cohorts.
      - A face is colored significant if any axis has `p < 0.05`.
      - Output: `cohort_comparison_<cohort1>_<cohort2>.gif`.
    - Welch/independent t-test per face:
      - `ttest_ind` per axis (`x`, `y`, `z`) between cohorts.
      - A face is colored significant if any axis has `p < 0.05`.
      - Output: `ttest_comparison_<cohort1>_<cohort2>.gif`.

- `--method per_subject`:
  - Still runs registration for both cohorts first.
  - Chooses one cohort as reference distribution (`--ref_distribution`) and scores each subject from the other cohort.
  - For each query subject:
    - Rigidly aligns query subject + all reference subjects (GPA).
    - For every vertex index, models the reference 3D coordinate distribution with covariance/precision.
    - Computes vertex Mahalanobis distance:
      - `d^2 = (x - mu)^T * Precision * (x - mu)`.
    - Converts to face score:
      - `face_mahal = sqrt(mean(d^2 of the face vertices))`.
  - Summary metrics:
    - `mahalanobis_per_face.csv`: per subject, per face Mahalanobis distance.
    - `mahalanobis_area_gt_5.csv`: percentage of mesh surface area with `face_mahal > 5`.
      - `area_percent = 100 * area(face_mahal > 5) / total_mesh_area`.
  - Outlier retry logic:
    - If both area and face-count exceed 40% above threshold (`>5`), the pipeline re-extracts/re-registers that subject with progressively tighter extraction margins and recomputes.
  - Visualization:
    - `<subject>_to_<ref_distribution>_mahalanobis.gif` heatmaps per subject.

`one_cohort_statistics`:

- Runs registration for the cohort, then rigid GPA alignment across its subjects.
- Computes per-face variability from face centroids:
  - For each face and subject, compute centroid `(x, y, z)`.
  - Compute standard deviation across subjects for each axis -> `std_face = (std_x, std_y, std_z)`.
  - GIF scalar map uses `||std_face||` as the variability intensity.
- Output:
  - `<cohort_name>_std.gif`.
- `--upper_limit` can clamp the colormap upper bound for easier cross-run visual comparison.

`visualize_subject`:

- Does not run Snakemake.
- Loads existing `head_affine_registered.obj` files, aligns the selected subject with reference cohort meshes, and launches an interactive Procrustes viewer (subject, cohort mean, and peer overlays).

`generate_full_body`:

- Runs `mouse_full_body.smk` then `extract_head_from_full_body.smk`.
- Produces full-body reconstruction intermediates and extracted head CT inputs for later skull registration/statistical commands.
- This command itself does not compute cohort statistics; it prepares data for downstream analysis.

## Platform Requirements

- OS: Linux is the intended runtime for Snakemake workflows
- Python: 3.9+
- Shell tools: `xvfb-run` for headless Slicer mesh transforms
- External executables: `snakemake`, 3D Slicer (configured by `slicer_core`)

Python dependencies are listed in `requirements.txt`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Full File Requirements for `main.py`

Required repository files:
- `main.py`
- `smk_configs/mouse_skull.yaml`
- `snakemake_workflows/generic_coregistration.smk`
- `snakemake_workflows/mouse_full_body.smk`
- `snakemake_workflows/extract_head_from_full_body.smk`
- `support/convert_mesh.py`
- `mouse_skull_pipelines/scripts/`

### Config Keys (`smk_configs/mouse_skull.yaml`)

Required for `main.py` command paths:
- `cohort_dir`
- `ref_image_dir`
- `parameter_file_nonlin`
- `parameter_file_lin`
- `mesh_transform_script`
- `slicer_core`
- `ref_im`
- `ref_mask`
- `body_part_detection_top_model`
- `male_face_perc`
- `female_face_perc`
- `coh_file`
- `control_points_model`
- `otsu_remove_outer`
- `linreg_simple`

Optional in current `main.py` path:
- `ref_landmarks`

Note on `ref_landmarks`:
- `compare_cohorts`, `one_cohort_statistics`, and `generate_full_body` do not actively use landmarks in their current workflow code path.
- It is still used by other workflow code in the repository (for example parts of `mouse_skull_analysis.smk`), so keep it in shared configs if you run those workflows.

### Input Layout Requirements

For `compare_cohorts` and `one_cohort_statistics`:

```text
<cohort_dir>/
  <subject_1>/
    ct_image.nii.gz
  <subject_2>/
    ct_image.nii.gz
  ...
```

Reference directory (`--reference_mouse`) should include:

```text
<reference_mouse>/
  oriented_image.nii.gz
  bone_coregistration_mask.nii.gz
  oriented_mesh.obj
  ref_landmarks.pts        # optional for main.py path, useful for other workflows
```

For `visualize_subject`:
- input 1: `<subject_dir>/head_affine_registered.obj` or direct mesh path
- input 2: reference cohort containing multiple `head_affine_registered.obj` meshes

For `generate_full_body` (raw source):
- subject folders are expected under cohort root
- each subject must include a `*Rec` folder with `.bmp` slices and `.log` metadata used by `mouse_full_body.smk`

Generated intermediate layout:

```text
<cohort_root>/
  full_body_workdir/
    <subject>/
      full_body_image_raw.nii.gz
      full_body_top_classes.xml
  mouse_skull_workdir/
    <subject>/
      ct_image.nii.gz
```

## CLI Reference (`main.py`)

General form:

```bash
python main.py <command> [options]
```

### `compare_cohorts`

Required options:
- `--cohort_1`
- `--cohort_2`
- `--reference_mouse`
- `--result_dir`

Optional:
- `--method {group,per_subject}` (default `group`)
- `--ref_distribution {cohort_1,cohort_2}` (default `cohort_2`, per-subject mode)
- `--fixate_plotting_max <float>` (per-subject mode color scale clamp)

Examples:

```bash
python main.py compare_cohorts \
  --cohort_1 /path/to/cohort_A \
  --cohort_2 /path/to/cohort_B \
  --reference_mouse /path/to/reference_mouse \
  --result_dir /path/to/results
```

```bash
python main.py compare_cohorts \
  --cohort_1 /path/to/cohort_A \
  --cohort_2 /path/to/cohort_B \
  --reference_mouse /path/to/reference_mouse \
  --result_dir /path/to/results \
  --method per_subject \
  --ref_distribution cohort_2 \
  --fixate_plotting_max 8.0
```

### `one_cohort_statistics`

Required options:
- `--cohort`
- `--reference_mouse`
- `--result_dir`

Optional:
- `--upper_limit <float>`

Example:

```bash
python main.py one_cohort_statistics \
  --cohort /path/to/cohort \
  --reference_mouse /path/to/reference_mouse \
  --result_dir /path/to/results \
  --upper_limit 2.5
```

### `visualize_subject`

Positional args:
- `subject` (subject folder or `head_affine_registered.obj`)
- `reference_cohort` (cohort with registered meshes)

Example:

```bash
python main.py visualize_subject \
  /path/to/subject_or_obj \
  /path/to/reference_cohort
```

### `generate_full_body`

Required:
- `--cohort-dir`

Optional:
- `--ref-image-dir`
- `--cores`
- `--keep-going`
- `--config-override KEY=VALUE` (repeatable)

Example:

```bash
python main.py generate_full_body \
  --cohort-dir /path/to/cohort_root \
  --config-override ref_image_dir=/path/to/reference_mouse \
  --config-override body_part_detection_top_model=/path/to/top_model.pth
```

Implementation note:
- current `main.py` hardcodes `cores=9` and `keep_going=True` when calling `prepare_full_body_and_extract_heads(...)`

## Outputs

Registration outputs per subject:
- `head_affine_registered.obj`
- `affine_head.txt`
- `affine_head.h5`
- `coregistered_head.nii.gz`
- `nl_transformation_params.txt`

`compare_cohorts` in `group` mode:
- `cohort_1_mean.obj`
- `cohort_2_mean.obj`
- `mean_diff_<cohort1>_<cohort2>.gif`
- `cohort_comparison_<cohort1>_<cohort2>.gif`
- `ttest_comparison_<cohort1>_<cohort2>.gif`

`compare_cohorts` in `per_subject` mode:
- `<result_dir>/<query_cohort>/<subject>_to_<ref_distribution>_mahalanobis.gif`
- `<result_dir>/<query_cohort>/mahalanobis_area_gt_5.csv`
- `<result_dir>/<query_cohort>/mahalanobis_per_face.csv`

`one_cohort_statistics`:
- `<cohort_name>_std.gif`

## Operational Notes

- Registration workflow cleanup keeps only a limited whitelist of files after run.
- Per-subject comparison caches Mahalanobis results in CSV and reuses cached values.
- Severe outliers can trigger automatic re-extraction/re-registration retries with reduced extraction margins.

## Troubleshooting

- `snakemake` missing: install in active environment and verify on PATH.
- Slicer transform failures: verify `slicer_core`, `mesh_transform_script`, and `xvfb-run`.
- Empty cohort mesh set: confirm each subject has `head_affine_registered.obj`.
- Missing config keys: check `smk_configs/mouse_skull.yaml` against the required list above.
