from pathlib import Path
from typing import Dict


def collect_registered_meshes(cohort_dir: Path) -> Dict[str, Path]:
    cohort_dir = Path(cohort_dir)
    if not cohort_dir.exists():
        raise FileNotFoundError(f"Cohort directory not found: {cohort_dir}")
    meshes: Dict[str, Path] = {}
    for entry in sorted(cohort_dir.iterdir()):
        if not entry.is_dir():
            continue
        mesh_path = entry / "head_affine_registered.obj"
        if mesh_path.exists():
            meshes[entry.name] = mesh_path
    if not meshes:
        raise FileNotFoundError(f"No 'head_affine_registered.obj' files found under {cohort_dir}")
    return meshes
