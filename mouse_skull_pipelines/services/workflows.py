import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from mouse_skull_pipelines.config.runtime import _repo_root, apply_config_overrides, load_yaml_mapping


def _log_duration(start_ts: float, label: str, subject: Optional[str] = None) -> None:
    elapsed = time.time() - start_ts
    if subject:
        print(f"[INFO] {label} done subject={subject} elapsed={elapsed:.1f}s")
    else:
        print(f"[INFO] {label} done elapsed={elapsed:.1f}s")


def clean_after_registration(cohort_folder: Path):
    files_to_leave = [
        "head_affine_registered.obj",
        "4_inv_mesh.flag",
        "ct_image.nii.gz",
        "nl_transformation_params.txt",
        "affine_head.txt",
        "coregistered_head.nii.gz",
        "affine_head.h5",
    ]

    for root, _, files in os.walk(cohort_folder):
        for file in files:
            if file not in files_to_leave:
                file_path = os.path.join(root, file)
                print(f"Removing {file_path}")
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    pass


def registration_to_reference(cohort_folder: Path, reference_mouse):
    registration_cfg = _repo_root() / "smk_configs" / "mouse_skull.yaml"
    script_home = _repo_root() / "snakemake_workflows" / "generic_coregistration.smk"

    config = load_yaml_mapping(registration_cfg)
    cohort_folder = Path(cohort_folder).expanduser().resolve()
    reference_mouse = Path(reference_mouse).expanduser().resolve()
    config["ref_image_dir"] = str(reference_mouse)
    config["cohort_dir"] = str(cohort_folder)

    try:
        run_snakemake_workflow(script_home, config, cores=9, keep_going=True)
        clean_after_registration(cohort_folder)
    except subprocess.CalledProcessError as e:
        print(f"Error during registration: {e}")
        print("Please check the config file and the cohort folder.")


def run_snakemake_workflow(snakefile: Path, config: Dict, cores: int, keep_going: bool) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        yaml.safe_dump(config, temp_file)
        temp_path = temp_file.name
    cmd = [
        "snakemake",
        "--snakefile",
        str(snakefile),
        "--configfile",
        temp_path,
        "--cores",
        str(cores),
    ]
    if keep_going:
        cmd.append("--keep-going")
    print(f"[INFO] Running snakemake: {snakefile} cores={cores} keep_going={keep_going}")
    print(f"[INFO] Snakemake config: {temp_path}")
    print(f"[INFO] Snakemake cmd: {' '.join(cmd)}")
    start_ts = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_ts
        print(f"[INFO] Snakemake finished in {elapsed:.1f}s: {snakefile}")
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


def prepare_full_body_and_extract_heads(
    cohort_dir: Path,
    ref_image_dir: Optional[Path],
    overrides: Optional[List[str]],
    cores: int,
    keep_going: bool,
) -> None:
    base_dir = _repo_root()
    config_path = base_dir / "smk_configs" / "mouse_skull.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_yaml_mapping(config_path)
    workflows_root = base_dir / "snakemake_workflows"
    cohort_dir = cohort_dir.expanduser().resolve()
    if not cohort_dir.exists():
        raise FileNotFoundError(f"Cohort directory not found: {cohort_dir}")
    config["cohort_dir"] = str(cohort_dir)

    apply_config_overrides(config, overrides)
    required_keys = {
        "cohort_dir",
        "ref_image_dir",
        "parameter_file_nonlin",
        "parameter_file_lin",
        "mesh_transform_script",
        "slicer_core",
        "ref_im",
        "ref_mask",
        "body_part_detection_top_model",
    }
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(
            "Missing required config keys for full body workflow: " + ", ".join(sorted(missing))
        )
    run_snakemake_workflow(workflows_root / "mouse_full_body.smk", config, cores, keep_going)
    run_snakemake_workflow(workflows_root / "extract_head_from_full_body.smk", config, cores, keep_going)
