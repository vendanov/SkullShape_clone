Project: Mouse Skull Pipelines

Purpose
- Automated processing and analysis of mouse skull micro‑CT data.
- Registers cohort CT volumes to a common reference, transforms meshes, and enables cohort‑level statistics and visualizations (e.g., TMD/BMD, group differences).

Key Capabilities
- Linear and non‑rigid registration of CT heads to a reference (NiftyReg reg_aladin, ANTs/Elastix; driven via Snakemake).
- Masking via Otsu thresholding; robust flipping/affine correction helpers for misoriented scans.
- Mesh handling: inverse‑warps reference mesh into subject space; Procrustes/GPA alignment; mean mesh computation (VTK).
- Cohort analysis and visualization: KS tests, t‑tests, Mahalanobis distance maps; animated GIFs over 3D meshes.
- Optional deep‑learning components for landmark/ROI detection (PyTorch/TensorFlow models used by some workflows).

Tech Stack
- Python, Snakemake, VTK, SimpleITK, nibabel, NumPy/SciPy, scikit‑learn, matplotlib.
- External tools/binaries: NiftyReg (`reg_aladin`), Elastix, ANTs, 3D Slicer (for mesh warping via Python script).
- ML: PyTorch/TorchVision (landmarks/ROIs), TensorFlow/Keras (legacy TMD/BMD scripts).

Repository Layout (selected)
- `snakemake_workflows/` — Orchestrates pipelines
  - `snakemake_workflows/generic_coregistration.smk` — Core CT→reference registration (Otsu → linear → non‑rigid → inverse‑mesh)
  - `snakemake_workflows/mouse_skull_analysis.smk` — End‑to‑end skull analysis utilities (conversion, ROI, mesh ops)
  - `snakemake_workflows/tmd_bmd.smk`, `tmd_bmd.py` — TMD/BMD processing (legacy/experimental)
- `mouse_skull_pipelines/scripts/` — Image/mesh algorithms and ML helpers
  - `scripts/image_algorithms.py` — Otsu, linear reg, elastix wrapper, ANTs helpers, flipping/affine utils
  - `scripts/mesh_algorithms.py`, `scripts/vtk_helper.py` — Mesh IO and VTK utilities
  - `scripts/gpa.py` — Generalized Procrustes alignment and mean mesh
  - `scripts/nets.py` — Landmark/ROI detection models and helpers
- `smk_configs/mouse_skull.yaml` — Configuration used by Snakemake workflows
- `support/convert_mesh.py` — Mesh transformation via 3D Slicer
- `runner_scripts/` — Convenience entry points for common runs
- `main.py` — CLI for registration via Snakemake and cohort comparisons/visualizations
- `models/` — Registration parameter files and model references
- `fancy_tmd_bmd/`, `notebooks/`, `matlab/` — Optional experiments, analysis, and legacy assets
- `slurm/` — Place generic SLURM sbatch scripts here (e.g., sbatch_registration.sh, sbatch_compare_cohorts.sh)

Data and Folder Conventions
- Cohort directory: contains per‑subject subfolders with input CT and outputs
  - Input per subject: `ct_image.nii.gz`
  - Outputs (produced incrementally by Snakemake):
    - `1_otsu.flag` → `bone_coregistration_mask.nii.gz`
    - `2_lin_cor.flag` → `coregistered_head.nii.gz`, `rigid_head.txt`, `affine_head.txt`
    - `3_nonrig.flag` → `warped_elastic.nii.gz`, `nl_transformation_params.txt`, `deformationField.h5`
    - `4_inv_mesh.flag` → `head_affine_registered.obj` (reference mesh transformed)
- Reference image directory: holds reference assets expected by configs
  - `oriented_image.nii.gz` (ref CT), `bone_coregistration_mask.nii.gz`
  - `oriented_mesh.obj` (ref mesh), `ref_landmarks.pts`
  - Optional ROI XMLs: e.g., `oriented_image_top.xml`, `oriented_image_side.xml`

Configuration
- Primary config: `smk_configs/mouse_skull.yaml`
  - `cohort_dir`: Path to the cohort root
  - `ref_image_dir`: Path to the reference assets folder
  - `parameter_file_lin`, `parameter_file_nonlin`: Parameter/weights for linear/nonlinear steps
  - `mesh_transform_script`: Path to `support/convert_mesh.py` (used by 3D Slicer)
  - `slicer_core`: Path to Slicer executable
  - `ref_im`, `ref_mask`, `ref_landmarks`: Filenames within the reference folder
  - Optional: model paths for landmark/ROI detection (`control_points_model`, `skull_rois_model`, `sutures_model`, etc.)
  - Flags: `linreg_simple`, `otsu_remove_outer`
  - Nonlinear tuning: adjust `nonrigid_registration.elastix_overrides` (e.g. `FinalGridSpacingInVoxels`, `MaximumNumberOfIterations`) to trade deformation strength vs. runtime.

Core Workflow: Generic Coregistration
- Location: `snakemake_workflows/generic_coregistration.smk`
- Steps
  - Otsu mask: thresholds `ct_image.nii.gz` → `bone_coregistration_mask.nii.gz`
  - Linear registration: rigid/affine via `reg_aladin` (with optional flips if metric low)
  - Non‑rigid registration: ANTs/Elastix produces `deformationField.h5` and warped image
  - Inverse mesh transform: uses 3D Slicer + `support/convert_mesh.py` to warp `oriented_mesh.obj` → subject space
- Driven by Snakemake `config` values for directories and tool paths.

CLI Entry Points
- File: `main.py`
- Commands
  - `compare_cohorts`
    - Args: `--cohort_1`, `--cohort_2`, `--reference_mouse`, `--result_dir`
    - Options: `--method {group|per_subject}` (default group), `--ref_distribution {cohort_1|cohort_2}` for per‑subject, `--fixate_plotting_max`
    - Performs mesh‑based comparisons (KS, t‑tests, mean diffs), generates visual summaries (GIFs)
  - `one_cohort_statistics`
    - Args: `--cohort`, `--reference_mouse`, `--result_dir`, `--upper_limit`
    - Computes internal variance/dispersion statistics and visualizations
- Programmatic function
  - `registration_to_reference(cohort_folder, reference_mouse)` spawns Snakemake run of `generic_coregistration.smk` using a temporary YAML built from `smk_configs/mouse_skull.yaml`.

How To Run (typical)
- Prepare data
  - Organize `cohort_dir` with subfolders, each containing `ct_image.nii.gz`.
  - Prepare `ref_image_dir` with `oriented_image.nii.gz`, `bone_coregistration_mask.nii.gz`, `oriented_mesh.obj`, `ref_landmarks.pts`.
- Adjust config
  - Edit `smk_configs/mouse_skull.yaml` to point to your local paths and tools.
- Run registration (via CLI)
  - `python main.py` through `registration_to_reference(...)` in your own script, or
  - Direct Snakemake: `snakemake --snakefile snakemake_workflows/generic_coregistration.smk --configfile smk_configs/mouse_skull.yaml --cores 8 --keep-going`
- Compare cohorts
  - `python main.py compare_cohorts --cohort_1 <path> --cohort_2 <path> --reference_mouse <ref_dir> --result_dir <out>`
  - Per‑subject option highlights deviations using Mahalanobis distances.

Dependencies
- Python packages: see `requirements.txt` (NumPy, SciPy, scikit‑learn, VTK, SimpleITK, torch/torchvision, nibabel, matplotlib, Pillow, tqdm, snakemake, etc.).
- External binaries (install separately and ensure on PATH or referenced in config):
  - NiftyReg (`reg_aladin`) for linear registration
  - Elastix (and/or ANTs) for non‑rigid registration
  - 3D Slicer (headless options) for mesh warping via `support/convert_mesh.py`

Assumptions and Conventions
- Subject folders are filtered by presence of `ct_image.nii.gz`.
- Registration quality check uses a correlation/metric threshold; automatic flips are attempted if low.
- Flags (`*.flag`) are used to checkpoint steps in Snakemake.
- Results and intermediate files are created next to each subject folder; optional clean‑up is available (`clean_after_registration`).

Extensibility
- Add new Snakemake rules under `snakemake_workflows/` and reference them from a config file.
- Extend mesh/statistics logic in `mouse_skull_pipelines/scripts/` (e.g., new metrics or visual encodings).
- Swap or tune parameter files in `models/` and model paths in `smk_configs/mouse_skull.yaml`.

Useful Filepaths (clickable)
- snakemake_workflows/generic_coregistration.smk
- snakemake_workflows/mouse_skull_analysis.smk
- mouse_skull_pipelines/scripts/image_algorithms.py
- mouse_skull_pipelines/scripts/gpa.py
- main.py
- smk_configs/mouse_skull.yaml
- support/convert_mesh.py
- requirements.txt

Notes
- The repository contains experimental notebooks and legacy scripts; core production flow is centered on `generic_coregistration.smk` and the `main.py` CLI.
- Paths in `smk_configs/mouse_skull.yaml` are environment‑specific and may require Windows vs. Linux adjustments.
- `python main.py visualize_subject <subject_dir> <reference_cohort>` opens an interactive VTK viewer aligning the subject with the cohort Procrustes mean (toggle peers with keyboard shortcuts).


Scripts designed to work on linux
