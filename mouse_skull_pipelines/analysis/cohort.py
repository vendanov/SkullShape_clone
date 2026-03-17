import csv
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import vtk

from mouse_skull_pipelines.analysis.mesh_stats import (
    compute_2cohort_differences,
    compute_and_save_mean_mesh,
    compute_facewise_centroid_std,
    read_vtk_polydata_points_faces,
)
from mouse_skull_pipelines.config.runtime import _repo_root, load_yaml_mapping
from mouse_skull_pipelines.services.workflows import (
    _log_duration,
    clean_after_registration,
    registration_to_reference,
    run_snakemake_workflow,
)
from mouse_skull_pipelines.scripts.gpa import gpa_new
from mouse_skull_pipelines.scripts.mesh_algorithms import read_mesh_from_file
from mouse_skull_pipelines.visualization.gifs import (
    produce_gif_coh_comparison,
    produce_gif_mean_diff,
    produce_gif_scalar_map,
    produce_gif_ttest_comparison,
    uproduce_gif_std,
)
def should_recompute(
    area_percent: float,
    percent_points_over: float,
    area_percent_threshold: float = 40.0,
    face_percent_threshold: float = 40.0,
) -> bool:
    return (
        area_percent >= area_percent_threshold
        and percent_points_over >= face_percent_threshold
    )


def _rmtree_with_retries(path: Path, retries: int = 6, base_delay_s: float = 0.2) -> None:
    # Some network/filesystem backends can transiently report "Directory not empty"
    # during recursive deletes; retry with short backoff before giving up.
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay_s * (attempt + 1))


def _find_cohort_root_for_extraction(cohort_dir: Path) -> Optional[Path]:
    candidates = [cohort_dir, cohort_dir.parent, cohort_dir.parent.parent]
    for cand in candidates:
        if (cand / "full_body_workdir").exists():
            return cand
    return None


def _force_recompute_subjects(subjects: List[str],
                              cohort_dir: Path,
                              reference_mouse: Path,
                              y_start_margin: int) -> bool:
    cohort_root = _find_cohort_root_for_extraction(cohort_dir)
    if cohort_root is None:
        print(f"[WARN] Cannot recompute subjects: full_body_workdir not found for {cohort_dir}")
        return False

    base_skull_dir = cohort_root / "mouse_skull_workdir"
    for subject in subjects:
        base_subject_dir = base_skull_dir / subject
        base_ct = base_subject_dir / "ct_image.nii.gz"
        base_ct.unlink(missing_ok=True)

    extract_smk = _repo_root() / "snakemake_workflows" / "extract_head_from_full_body.smk"
    extract_cfg = {
        "cohort_dir": str(cohort_root),
        "subjects": subjects,
        "y_start_margin": int(y_start_margin),
    }
    run_snakemake_workflow(extract_smk, extract_cfg, cores=9, keep_going=True)

    if base_skull_dir in cohort_dir.parents and cohort_dir != base_skull_dir:
        for subject in subjects:
            base_subject_dir = base_skull_dir / subject
            dst_dir = cohort_dir / subject
            if dst_dir.exists():
                _rmtree_with_retries(dst_dir)
            shutil.copytree(base_subject_dir, dst_dir, copy_function=shutil.copy2, symlinks=True)

    for subject in subjects:
        subject_dir = cohort_dir / subject
        for fname in [
            "1_otsu.flag",
            "2_lin_cor.flag",
            "3_nonrig.flag",
            "4_inv_mesh.flag",
            "head_affine_registered.obj",
            "bone_coregistration_mask.nii.gz",
            "coregistered_head.nii.gz",
            "nonlin_image.nii.gz",
            "nl_transformation_params.txt",
        ]:
            (subject_dir / fname).unlink(missing_ok=True)

    config = load_yaml_mapping(_repo_root() / "smk_configs" / "mouse_skull.yaml")
    config["ref_image_dir"] = str(reference_mouse)
    config["cohort_dir"] = str(cohort_dir)
    config["subjects"] = subjects
    reg_smk = _repo_root() / "snakemake_workflows" / "generic_coregistration.smk"
    run_snakemake_workflow(reg_smk, config, cores=9, keep_going=True)
    clean_after_registration(cohort_dir)
    return True


def _compute_mahalanobis_for_subject(subject: str,
                                     mesh_path: Path,
                                     ref_cohort_meshes: Dict[str, Path],
                                     area_threshold: float,
                                     use_cache: bool = True) -> Tuple[np.ndarray, vtk.vtkPolyData, float]:
    gpa_input = dict(ref_cohort_meshes)
    gpa_input[subject] = mesh_path
    gpa_all = gpa_new(gpa_input, modes="rigid", use_cache=use_cache)

    aligned_key_mesh = gpa_all[subject]
    aligned_ref_meshes = [gpa_all[k] for k in gpa_all.keys() if k != subject]

    pts_key, faces = read_vtk_polydata_points_faces(aligned_key_mesh)

    from sklearn.covariance import EmpiricalCovariance
    ref_points_stack = np.stack(
        [read_vtk_polydata_points_faces(m)[0] for m in aligned_ref_meshes], axis=0
    )
    n_ref, n_points, _ = ref_points_stack.shape

    mu_pts = np.zeros((n_points, 3), dtype=np.float64)
    prec_pts = np.zeros((n_points, 3, 3), dtype=np.float64)
    eps = 1e-6
    for p in range(n_points):
        X = ref_points_stack[:, p, :]
        if n_ref >= 2:
            emp = EmpiricalCovariance()
            emp.fit(X)
            md = emp.mahalanobis(X)
            thr = np.percentile(md, 99.0)
            keep = md < thr
            if keep.sum() >= 2:
                X = X[keep]
                emp.fit(X)
                mu_pts[p] = emp.location_
                prec_pts[p] = emp.precision_
                continue
        mu_p = X.mean(axis=0)
        C = np.cov(X, rowvar=False) if n_ref > 1 else np.eye(3) * eps
        C = np.asarray(C, dtype=np.float64)
        C.flat[::4] += eps
        mu_pts[p] = mu_p
        prec_pts[p] = np.linalg.pinv(C)

    deltas = pts_key - mu_pts
    d2_pts = np.einsum('pi,pij,pj->p', deltas, prec_pts, deltas, optimize=True)
    d2_pts = np.clip(d2_pts, 0.0, None)
    mahal = np.sqrt(d2_pts[faces].mean(axis=1))

    tri_pts = pts_key[faces]
    vec1 = tri_pts[:, 1, :] - tri_pts[:, 0, :]
    vec2 = tri_pts[:, 2, :] - tri_pts[:, 0, :]
    face_areas = 0.5 * np.linalg.norm(np.cross(vec1, vec2), axis=1)
    total_area = float(face_areas.sum())
    area_over = float(face_areas[mahal > area_threshold].sum())
    area_percent = 100.0 * area_over / total_area if total_area > 0 else 0.0

    return mahal, aligned_key_mesh, area_percent


def _compute_mahalanobis_from_aligned_meshes(subject: str,
                                             aligned_meshes: Dict[str, vtk.vtkPolyData],
                                             area_threshold: float) -> Tuple[np.ndarray, float]:
    aligned_key_mesh = aligned_meshes[subject]
    aligned_ref_meshes = [mesh for key, mesh in aligned_meshes.items() if key != subject]
    if not aligned_ref_meshes:
        raise ValueError(f"Need at least one reference mesh to compute Mahalanobis for {subject}.")

    pts_key, faces = read_vtk_polydata_points_faces(aligned_key_mesh)
    ref_points_stack = np.stack(
        [read_vtk_polydata_points_faces(mesh)[0] for mesh in aligned_ref_meshes],
        axis=0,
    )
    n_ref, n_points, _ = ref_points_stack.shape

    mu_pts = np.zeros((n_points, 3), dtype=np.float64)
    prec_pts = np.zeros((n_points, 3, 3), dtype=np.float64)
    eps = 1e-6

    for p in range(n_points):
        X = ref_points_stack[:, p, :]
        mu_p = X.mean(axis=0)
        C = np.cov(X, rowvar=False) if n_ref > 1 else np.eye(3) * eps
        C = np.asarray(C, dtype=np.float64)
        C.flat[::4] += eps
        mu_pts[p] = mu_p
        prec_pts[p] = np.linalg.pinv(C)

    deltas = pts_key - mu_pts
    d2_pts = np.einsum('pi,pij,pj->p', deltas, prec_pts, deltas, optimize=True)
    d2_pts = np.clip(d2_pts, 0.0, None)
    mahal = np.sqrt(d2_pts[faces].mean(axis=1))

    tri_pts = pts_key[faces]
    vec1 = tri_pts[:, 1, :] - tri_pts[:, 0, :]
    vec2 = tri_pts[:, 2, :] - tri_pts[:, 0, :]
    face_areas = 0.5 * np.linalg.norm(np.cross(vec1, vec2), axis=1)
    total_area = float(face_areas.sum())
    area_over = float(face_areas[mahal > area_threshold].sum())
    area_percent = 100.0 * area_over / total_area if total_area > 0 else 0.0
    return mahal, area_percent


def _percent_over_threshold(mahal: np.ndarray, threshold: float) -> float:
    if mahal.size == 0:
        return 0.0
    return 100.0 * float(np.sum(mahal > threshold)) / float(len(mahal))


def _collect_registered_meshes(cohort_dir: Path) -> Dict[str, Path]:
    # Map subject folder names to registered mesh paths.
    return {
        x.name: x / "head_affine_registered.obj"
        for x in Path(cohort_dir).iterdir()
        if (x / "head_affine_registered.obj").exists()
    }


def _select_reference_distribution(cohort_1: Path,
                                   cohort_2: Path,
                                   result_dir: Path,
                                   ref_distribution: str,
                                   cohort_1_meshes: Dict[str, Path],
                                   cohort_2_meshes: Dict[str, Path]):
    # Choose which cohort defines the reference distribution.
    if ref_distribution == "cohort_1":
        ref_cohort_meshes = cohort_1_meshes
        query_cohort_meshes = cohort_2_meshes
        res_dir_new = result_dir / cohort_2.name
        query_cohort_dir = cohort_2
    else:
        ref_cohort_meshes = cohort_2_meshes
        query_cohort_meshes = cohort_1_meshes
        res_dir_new = result_dir / cohort_1.name
        query_cohort_dir = cohort_1
    return ref_cohort_meshes, query_cohort_meshes, res_dir_new, query_cohort_dir


def _load_mahalanobis_cache(area_csv: Path,
                            mahal_csv: Path) -> Tuple[Dict[str, float],
                                                     Dict[str, Dict[int, float]],
                                                     bool]:
    # Load cached per-subject metrics when available.
    cached_area: Dict[str, float] = {}
    cached_mahal: Dict[str, Dict[int, float]] = {}
    write_cache = False
    if area_csv.exists() and mahal_csv.exists():
        with open(area_csv, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            area_col = None
            if reader.fieldnames:
                if "area_percent_mahal_gt_threshold" in reader.fieldnames:
                    area_col = "area_percent_mahal_gt_threshold"
                elif "area_percent_mahal_gt_5" in reader.fieldnames:
                    area_col = "area_percent_mahal_gt_5"
                else:
                    for candidate in reader.fieldnames:
                        if candidate.startswith("area_percent_mahal_gt_"):
                            area_col = candidate
                            break
            if area_col is None:
                raise ValueError(
                    f"Area CSV {area_csv} missing an area_percent_mahal_gt_* column."
                )
            for row in reader:
                subject = row["subject"]
                area_val = float(row[area_col])
                cached_area[subject] = area_val
        with open(mahal_csv, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                subject = row["subject"]
                face_idx = int(row["face_index"])
                val = float(row["mahalanobis"])
                cached_mahal.setdefault(subject, {})[face_idx] = val
    else:
        write_cache = True
    return cached_area, cached_mahal, write_cache


def _compute_subject_info(query_cohort_meshes: Dict[str, Path],
                          ref_cohort_meshes: Dict[str, Path],
                          cached_area: Dict[str, float],
                          cached_mahal: Dict[str, Dict[int, float]],
                          area_threshold: float) -> Tuple[Dict[str, Dict], bool]:
    # Build per-subject stats, using cache when possible.
    subject_info: Dict[str, Dict] = {}
    write_cache = False
    for key, mesh_path in query_cohort_meshes.items():
        cached_area_val = cached_area.get(key)
        cached_mahal_vals = cached_mahal.get(key)
        if cached_area_val is not None and cached_mahal_vals is not None:
            max_idx = max(cached_mahal_vals.keys()) if cached_mahal_vals else -1
            mahal = np.zeros(max_idx + 1, dtype=np.float64) if max_idx >= 0 else np.array([])
            for idx, val in cached_mahal_vals.items():
                mahal[idx] = val
            percent_over = _percent_over_threshold(mahal, area_threshold)
            subject_info[key] = {
                "source": "cached",
                "area_percent": cached_area_val,
                "percent_over": percent_over,
                "mahal": mahal,
                "aligned_mesh": None,
            }
            print(
                f"[INFO] Outlier check {key}: area_percent={cached_area_val:.2f}, "
                f"percent_over={percent_over:.2f} (cached)"
            )
        else:
            print(f"[INFO] Computing {key} (y_start_margin=40)")
            mahal, aligned_key_mesh, area_percent = _compute_mahalanobis_for_subject(
                key, mesh_path, ref_cohort_meshes, area_threshold
            )
            percent_over = _percent_over_threshold(mahal, area_threshold)
            subject_info[key] = {
                "source": "computed",
                "area_percent": area_percent,
                "percent_over": percent_over,
                "mahal": mahal,
                "aligned_mesh": aligned_key_mesh,
            }
            print(
                f"[INFO] Outlier check {key}: area_percent={area_percent:.2f}, "
                f"percent_over={percent_over:.2f} (computed)"
            )
            write_cache = True
    return subject_info, write_cache


def _remove_outlier_gifs(outliers: List[str],
                         res_dir_new: Path,
                         ref_distribution: str) -> None:
    # Remove cached GIFs for outliers to force regeneration.
    for subject in outliers:
        gif_path = res_dir_new / f"{subject}_to_{ref_distribution}_mahalanobis.gif"
        try:
            gif_path.unlink()
        except FileNotFoundError:
            pass


def _recompute_outliers(outliers: List[str],
                        query_cohort_meshes: Dict[str, Path],
                        ref_cohort_meshes: Dict[str, Path],
                        query_cohort_dir: Path,
                        reference_mouse: Path,
                        area_threshold: float,
                        subject_info: Dict[str, Dict],
                        recompute_area_percent_threshold: float,
                        recompute_face_percent_threshold: float) -> Tuple[set, bool]:
    # Retry extraction with tighter margins for persistent outliers.
    force_gif_subjects = set()
    write_cache = False
    if outliers:
        for margin in (30, 20, 10, 0):
            print(f"[INFO] Recomputing outliers with y_start_margin={margin}: {outliers}")
            if not _force_recompute_subjects(outliers, query_cohort_dir, reference_mouse, margin):
                break
            still_outliers = []
            for subject in list(outliers):
                mahal, aligned_key_mesh, area_percent = _compute_mahalanobis_for_subject(
                    subject, query_cohort_meshes[subject], ref_cohort_meshes, area_threshold,
                    use_cache=False
                )
                percent_over = _percent_over_threshold(mahal, area_threshold)
                print(
                    f"[INFO] Recompute check {subject}: area_percent={area_percent:.2f}, "
                    f"percent_over={percent_over:.2f}"
                )
                subject_info[subject] = {
                    "source": f"recomputed_{margin}",
                    "area_percent": area_percent,
                    "percent_over": percent_over,
                    "mahal": mahal,
                    "aligned_mesh": aligned_key_mesh,
                }
                force_gif_subjects.add(subject)
                if should_recompute(
                    area_percent,
                    percent_over,
                    recompute_area_percent_threshold,
                    recompute_face_percent_threshold,
                ):
                    still_outliers.append(subject)
            outliers = still_outliers
            write_cache = True
            if not outliers:
                break
            if margin == 0:
                print(f"[WARN] Outliers persist at y_start_margin=0: {outliers}")
    return force_gif_subjects, write_cache


def _append_mahalanobis_rows(subject_info: Dict[str, Dict],
                             area_rows: List[Dict],
                             mahal_rows: List[Dict]) -> None:
    # Expand per-subject info into CSV rows.
    for key, info in subject_info.items():
        area_rows.append({
            "subject": key,
            "area_percent_mahal_gt_threshold": info["area_percent"],
        })
        for face_idx, mahal_val in enumerate(info["mahal"]):
            mahal_rows.append({
                "subject": key,
                "face_index": face_idx,
                "mahalanobis": float(mahal_val),
            })


def _write_one_cohort_subject_summary(subject_info: Dict[str, Dict],
                                      summary_csv: Path,
                                      threshold: float) -> None:
    rows = []
    for subject, info in sorted(subject_info.items()):
        mahal = np.asarray(info.get("mahal", []), dtype=np.float64)
        rows.append({
            "subject": subject,
            "source": info.get("source", "computed"),
            "n_faces": int(mahal.size),
            "mean_mahalanobis": float(mahal.mean()) if mahal.size else 0.0,
            "median_mahalanobis": float(np.median(mahal)) if mahal.size else 0.0,
            "max_mahalanobis": float(mahal.max()) if mahal.size else 0.0,
            "percent_faces_mahal_gt_threshold": float(info.get("percent_over", 0.0)),
            "area_percent_mahal_gt_threshold": float(info.get("area_percent", 0.0)),
            "threshold": float(threshold),
        })

    with open(summary_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "subject",
                "source",
                "n_faces",
                "mean_mahalanobis",
                "median_mahalanobis",
                "max_mahalanobis",
                "percent_faces_mahal_gt_threshold",
                "area_percent_mahal_gt_threshold",
                "threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved one-cohort subject summary at {summary_csv}")


def _compute_one_cohort_subject_outliers(cohort_meshes: Dict[str, Path],
                                         threshold: float) -> Dict[str, Dict]:
    if len(cohort_meshes) < 2:
        print("[WARN] Need at least two registered subjects to compute one-cohort Mahalanobis outlier statistics.")
        return {}

    print(f"[INFO] Running one rigid GPA for {len(cohort_meshes)} cohort subjects")
    aligned_meshes = gpa_new(cohort_meshes, modes="rigid", use_cache=True)
    subject_info: Dict[str, Dict] = {}
    for subject in sorted(aligned_meshes.keys()):
        mahal, area_percent = _compute_mahalanobis_from_aligned_meshes(
            subject,
            aligned_meshes,
            threshold,
        )
        percent_over = _percent_over_threshold(mahal, threshold)
        subject_info[subject] = {
            "source": "computed",
            "area_percent": area_percent,
            "percent_over": percent_over,
            "mahal": mahal,
            "aligned_mesh": None,
        }
        print(
            f"[INFO] One-cohort outlier check {subject}: "
            f"area_percent={area_percent:.2f}, percent_over={percent_over:.2f}"
        )
    return subject_info


def _emit_subject_gifs(subject_info: Dict[str, Dict],
                       query_cohort_meshes: Dict[str, Path],
                       res_dir_new: Path,
                       ref_distribution: str,
                       fixate_plotting_max: Optional[float],
                       force_gif_subjects: set) -> None:
    # Generate per-subject GIFs with cache awareness.
    for key, mesh_path in query_cohort_meshes.items():
        info = subject_info.get(key)
        if info is None:
            continue
        mahal = info["mahal"]
        if mahal is None or len(mahal) == 0:
            continue
        gif_path = res_dir_new / f"{key}_to_{ref_distribution}_mahalanobis.gif"
        if info["aligned_mesh"] is not None:
            aligned_key_mesh = info["aligned_mesh"]
        else:
            aligned_key_mesh = read_mesh_from_file(mesh_path)
        recomputed = info.get("source") != "cached"
        if key in force_gif_subjects or recomputed or not gif_path.exists():
            produce_gif_scalar_map(aligned_key_mesh, mahal, str(gif_path),
                                   cmap_name='magma',
                                   clip_percentile=None if fixate_plotting_max is not None else 95.0,
                                   max_value=fixate_plotting_max,
                                   colorbar_label='Mahalanobis distance')
            print(f"Saved Mahalanobis distance GIF for {key} at {gif_path}")
        else:
            print(f"Skipping GIF for {key} (cached), already exists at {gif_path}")


def _write_mahalanobis_cache(area_rows: List[Dict],
                             mahal_rows: List[Dict],
                             area_csv: Path,
                             mahal_csv: Path) -> None:
    # Collapse duplicate rows and write cache CSVs.
    if area_rows:
        area_latest = {}
        for row in area_rows:
            area_latest[row["subject"]] = row["area_percent_mahal_gt_threshold"]
        area_rows[:] = [
            {"subject": subject, "area_percent_mahal_gt_threshold": area_val}
            for subject, area_val in area_latest.items()
        ]
    if mahal_rows:
        mahal_latest = {}
        for row in mahal_rows:
            key = (row["subject"], row["face_index"])
            mahal_latest[key] = row["mahalanobis"]
        mahal_rows[:] = [
            {"subject": subject, "face_index": face_idx, "mahalanobis": val}
            for (subject, face_idx), val in mahal_latest.items()
        ]
    if area_rows:
        with open(area_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["subject", "area_percent_mahal_gt_threshold"],
            )
            writer.writeheader()
            writer.writerows(area_rows)
        print(f"Saved Mahalanobis area summary at {area_csv}")
    if mahal_rows:
        with open(mahal_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["subject", "face_index", "mahalanobis"])
            writer.writeheader()
            writer.writerows(mahal_rows)
        print(f"Saved Mahalanobis per-face values at {mahal_csv}")


def create_mahalanobis_heatmap_from_csv(
    mahal_csv: Path,
    reference_mesh: Path,
    output_dir: Path,
    subject: Optional[str] = None,
    fixate_plotting_max: Optional[float] = None,
) -> None:
    mahal_csv = Path(mahal_csv).expanduser().resolve()
    reference_mesh = Path(reference_mesh).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not mahal_csv.exists():
        raise FileNotFoundError(f"Mahalanobis CSV not found: {mahal_csv}")
    if not reference_mesh.exists():
        raise FileNotFoundError(f"Reference mesh not found: {reference_mesh}")

    with open(mahal_csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        subject_col = "subject" if "subject" in fieldnames else (
            "subj_id" if "subj_id" in fieldnames else None
        )
        face_col = "face_index" if "face_index" in fieldnames else (
            "face_id" if "face_id" in fieldnames else None
        )
        if subject_col is None:
            raise ValueError("CSV must contain either 'subject' or 'subj_id'.")
        if face_col is None:
            raise ValueError("CSV must contain either 'face_index' or 'face_id'.")
        if "mahalanobis" not in fieldnames:
            raise ValueError("CSV must contain 'mahalanobis'.")

        per_subject = {}
        for row in reader:
            subj = str(row[subject_col]).strip()
            if not subj:
                continue
            face_idx = int(row[face_col])
            value = float(row["mahalanobis"])
            per_subject.setdefault(subj, {})[face_idx] = value

    if not per_subject:
        raise ValueError(f"No rows found in {mahal_csv}")

    if subject is not None and subject not in per_subject:
        subjects = ", ".join(sorted(per_subject.keys()))
        raise ValueError(
            f"Subject '{subject}' not found in CSV. Available subjects: {subjects}"
        )

    mesh = read_mesh_from_file(reference_mesh)
    _, faces = read_vtk_polydata_points_faces(mesh)
    n_faces = int(faces.shape[0])

    subjects_to_render = [subject] if subject is not None else sorted(per_subject.keys())
    output_dir.mkdir(parents=True, exist_ok=True)
    for subj in subjects_to_render:
        values = np.zeros(n_faces, dtype=float)
        for face_idx, val in per_subject[subj].items():
            if 0 <= face_idx < n_faces:
                values[face_idx] = val
        output_gif = output_dir / f"{subj}_mahalanobis.gif"
        produce_gif_scalar_map(
            mesh,
            values,
            str(output_gif),
            cmap_name='magma',
            clip_percentile=None if fixate_plotting_max is not None else 95.0,
            max_value=fixate_plotting_max,
            colorbar_label="Mahalanobis distance",
        )
        print(f"Saved Mahalanobis heatmap GIF for {subj} at {output_gif}")


def compare_cohorts_per_subject(cohort_1: str,
                                cohort_2: str,
                                reference_mouse: str,
                                result_dir: str,
                                ref_distribution: str = "cohort_2",
                                fixate_plotting_max: Optional[float] = None,
                                area_threshold: float = 5.0,
                                recompute_area_percent_threshold: float = 40.0,
                                recompute_face_percent_threshold: float = 40.0):
    """
    Per-subject comparison: for each mouse in one cohort, compute how far it is
    from the distribution of the other cohort.

    Creates rotating GIFs visualizing the Mahalanobis distance (per face) of
    each subject in the query cohort to the reference cohort's distribution.
    Returns a list of created GIF paths.
    """
    cohort_1 = Path(cohort_1)
    cohort_2 = Path(cohort_2)
    reference_mouse = Path(reference_mouse)
    result_dir = Path(result_dir)
    # Ensure output directory exists for saving GIFs
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"Per-subject comparison: {cohort_1} vs distribution of {cohort_2} (ref={ref_distribution})")

    # Preprocessing: registration (same as other commands)
    registration_to_reference(cohort_1, reference_mouse)
    registration_to_reference(cohort_2, reference_mouse)

    # Load meshes and pick which cohort serves as the reference distribution.
    cohort_1_meshes = _collect_registered_meshes(cohort_1)
    cohort_2_meshes = _collect_registered_meshes(cohort_2)
    print(f"Found meshes -> cohort_1: {len(cohort_1_meshes)}, cohort_2: {len(cohort_2_meshes)}")
    ref_cohort_meshes, query_cohort_meshes, res_dir_new, query_cohort_dir = (
        _select_reference_distribution(
            cohort_1, cohort_2, result_dir, ref_distribution, cohort_1_meshes, cohort_2_meshes
        )
    )
    res_dir_new.mkdir(parents=True, exist_ok=True)

    # Compute per-subject Mahalanobis against the reference distribution.

    area_rows = []
    mahal_rows = []
    threshold_label = str(area_threshold).replace(".", "p")
    area_csv = (
        res_dir_new / "mahalanobis_area_gt_5.csv"
        if area_threshold == 5.0
        else res_dir_new / f"mahalanobis_area_gt_{threshold_label}.csv"
    )
    mahal_csv = res_dir_new / "mahalanobis_per_face.csv"
    cached_area, cached_mahal, write_cache = _load_mahalanobis_cache(area_csv, mahal_csv)
    subject_info, computed_cache = _compute_subject_info(
        query_cohort_meshes, ref_cohort_meshes, cached_area, cached_mahal, area_threshold
    )
    write_cache = write_cache or computed_cache

    # Recompute outliers and flag them for GIF regeneration.
    outliers = [k for k, info in subject_info.items()
                if should_recompute(
                    info["area_percent"],
                    info["percent_over"],
                    recompute_area_percent_threshold,
                    recompute_face_percent_threshold,
                )]
    _remove_outlier_gifs(outliers, res_dir_new, ref_distribution)
    force_gif_subjects, recomputed_cache = _recompute_outliers(
        outliers,
        query_cohort_meshes,
        ref_cohort_meshes,
        query_cohort_dir,
        reference_mouse,
        area_threshold,
        subject_info,
        recompute_area_percent_threshold,
        recompute_face_percent_threshold,
    )
    write_cache = write_cache or recomputed_cache

    _append_mahalanobis_rows(subject_info, area_rows, mahal_rows)
    _emit_subject_gifs(
        subject_info,
        query_cohort_meshes,
        res_dir_new,
        ref_distribution,
        fixate_plotting_max,
        force_gif_subjects,
    )
    if write_cache:
        _write_mahalanobis_cache(area_rows, mahal_rows, area_csv, mahal_csv)


def compare_cohorts(cohort_1: str, cohort_2: str, reference_mouse: str, result_dir: str):
    """
    Compare two cohorts of mice and save the results to a file.
    """
    cohort_1 = Path(cohort_1)
    cohort_2 = Path(cohort_2)

    reference_mouse = Path(reference_mouse)
    result_dir = Path(result_dir)

    print(f"Comparing cohorts {cohort_1} and {cohort_2} with reference mouse {reference_mouse}")
    # Load the cohorts and reference mouse data

    registration_to_reference(cohort_1, reference_mouse)
    registration_to_reference(cohort_2, reference_mouse)
    # Load the cohorts and reference mouse data

    cohort_1_meshes = { x.name : x/"head_affine_registered.obj" for x in Path(cohort_1).iterdir()  if (x/"head_affine_registered.obj").exists() }
    cohort_2_meshes = { x.name : x/"head_affine_registered.obj" for x in Path(cohort_2).iterdir()  if (x/"head_affine_registered.obj").exists() }

    compute_and_save_mean_mesh(cohort_1_meshes, result_dir / "cohort_1_mean.obj")

    compute_and_save_mean_mesh(cohort_2_meshes, result_dir / "cohort_2_mean.obj")


    print(f"Number of meshes in cohort 1: {len(cohort_1_meshes)}")
    print(f"Number of meshes in cohort 2: {len(cohort_2_meshes)}")
    coh_aligned = gpa_new({**cohort_1_meshes, **cohort_2_meshes}, modes="rigid")

    print(f"Number of meshes in cohort al: {len(coh_aligned)}")

    # Save the aligned meshes to a file
    print("Saving the aligned meshes to a file",coh_aligned.keys())
    coh_1_meshes = { key : value for key, value in coh_aligned.items() if key in cohort_1_meshes }
    coh_2_meshes = { key : value for key, value in coh_aligned.items() if key in cohort_2_meshes }

    print(f"Number of meshes in cohort 1: {len(coh_1_meshes)}")
    print(f"Number of meshes in cohort 2: {len(coh_2_meshes)}")
    # Compute the statistics for the two cohorts

    ks_results = compute_2cohort_differences(coh_1_meshes, coh_2_meshes)

    # Save the results to a file
    def _read_mesh_from_file(mesh_file):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(str(mesh_file))
        reader.Update()
        return reader.GetOutput()

    mesh_file = next(iter(coh_1_meshes.values()))

    coh_1_name = cohort_1.name
    coh_2_name = cohort_2.name

    produce_gif_mean_diff(mesh_file, ks_results['mean_diff'], result_dir / f"mean_diff_{coh_1_name}_{coh_2_name}.gif")

    produce_gif_coh_comparison(mesh_file, ks_results['ks'], result_dir / f"cohort_comparison_{coh_1_name}_{coh_2_name}.gif")
    produce_gif_ttest_comparison(mesh_file,ks_results['t_test'], result_dir / f"ttest_comparison_{coh_1_name}_{coh_2_name}.gif")





def one_cohort_statistics(cohort: str,
                          reference_mouse: str,
                          result_dir: str,
                          upper_limit: Optional[float] = None,
                          mahal_summary_csv: Optional[str] = None,
                          mahal_threshold: float = 5.0,
                          skip_variability_gif: bool = False,
                          fast_outliers: bool = False):
    """
    Compute the statistics for one cohort of mice and save the results to a file.
    """
    cohort = Path(cohort)
    reference_mouse = Path(reference_mouse)
    result_dir = Path(result_dir)

    print(f"Computing statistics for cohort {cohort} with reference mouse {reference_mouse}")

    # Load the cohort and reference mouse data
    registration_to_reference(cohort, reference_mouse)
    # Load the cohort and reference mouse data

    cohort_meshes = { x.name : x/"head_affine_registered.obj" for x in Path(cohort).iterdir()  if (x/"head_affine_registered.obj").exists() }

    #compute_and_save_mean_mesh(cohort_meshes, result_dir / "cohort_mean.obj")

    print(f"Number of meshes in cohort: {len(cohort_meshes)}")
    if not cohort_meshes:
        raise ValueError(f"No registered meshes found in cohort: {cohort}")

    if not skip_variability_gif:
        coh_aligned = gpa_new(cohort_meshes, modes="rigid")
        std_results = compute_facewise_centroid_std(coh_aligned)
        print("std_results", std_results)
        mesh_file = next(iter(coh_aligned.values()))
        uproduce_gif_std(mesh_file,std_results, result_dir / f"{cohort.name}_std.gif", upper_limit=upper_limit)

    if mahal_summary_csv is not None:
        if fast_outliers:
            print("[INFO] Using fast approximate outlier mode (single cohort-wide GPA).")
            subject_info = _compute_one_cohort_subject_outliers(
                cohort_meshes=cohort_meshes,
                threshold=mahal_threshold,
            )
        else:
            print("[INFO] Using exact outlier mode (leave-one-out GPA per subject).")
            subject_info = {}
            for subject, mesh_path in sorted(cohort_meshes.items()):
                ref_meshes = {key: value for key, value in cohort_meshes.items() if key != subject}
                if not ref_meshes:
                    continue
                mahal, _, area_percent = _compute_mahalanobis_for_subject(
                    subject,
                    mesh_path,
                    ref_meshes,
                    mahal_threshold,
                    use_cache=True,
                )
                percent_over = _percent_over_threshold(mahal, mahal_threshold)
                subject_info[subject] = {
                    "source": "computed_exact",
                    "area_percent": area_percent,
                    "percent_over": percent_over,
                    "mahal": mahal,
                    "aligned_mesh": None,
                }
                print(
                    f"[INFO] Exact one-cohort outlier check {subject}: "
                    f"area_percent={area_percent:.2f}, percent_over={percent_over:.2f}"
                )
        if subject_info:
            area_rows: List[Dict] = []
            mahal_rows: List[Dict] = []
            _append_mahalanobis_rows(subject_info, area_rows, mahal_rows)
            summary_csv = Path(mahal_summary_csv)
            mahal_csv = result_dir / f"{cohort.name}_mahalanobis_per_face.csv"
            _write_one_cohort_subject_summary(subject_info, summary_csv, mahal_threshold)
            _write_mahalanobis_cache(
                area_rows,
                mahal_rows,
                result_dir / f"{cohort.name}_mahalanobis_area_summary.csv",
                mahal_csv,
            )






