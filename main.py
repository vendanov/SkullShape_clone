
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import scipy.io as scio
import numpy as np
import vtk
import yaml

import tempfile

from pathlib import Path

from matplotlib import pyplot as plt
from sympy.physics.quantum.identitysearch import scipy

from mouse_skull_pipelines.scripts.gpa import gpa_new, compute_mean_mesh
from mouse_skull_pipelines.scripts.mesh_algorithms import read_mesh_from_file

APP_VERSION = "mouse_skull_pipelines"

def _get_app_version() -> str:
    return APP_VERSION


def _log_duration(start_ts: float, label: str, subject: Optional[str] = None) -> None:
    elapsed = time.time() - start_ts
    if subject:
        print(f"[INFO] {label} done subject={subject} elapsed={elapsed:.1f}s")
    else:
        print(f"[INFO] {label} done elapsed={elapsed:.1f}s")

def clean_after_registration(cohort_folder: Path):
    """
    Clean the cohort folder after registration.
    """

    files_to_leave = ["head_affine_registered.obj", "4_inv_mesh.flag","ct_image.nii.gz",
                      "nl_transformation_params.txt" , "affine_head.txt", "coregistered_head.nii.gz",
                      "affine_head.h5"
                      ]

    # itearate over the files in the cohort folder and subfolders
    for root, dirs, files in os.walk(cohort_folder):
        for file in files:
            # if the file is not in the list of files to leave, remove it
            if file not in files_to_leave:
                file_path = os.path.join(root, file)
                print(f"Removing {file_path}")
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    pass


def resolve_curent_parent_dir():
    # current file path
    current_file = Path(__file__).resolve()
    parent_dir = current_file.parent
    return parent_dir



def registration_to_reference(cohort_folder: Path, reference_mouse):

    """
    Check if the registration is done for the given cohort folder.
    """

    current_file = Path(__file__).resolve()
    parent_dir = current_file.parent
    registration_cfg = parent_dir / 'smk_configs' / 'mouse_skull.yaml'
    script_home = parent_dir / 'snakemake_workflows' / 'generic_coregistration.smk'

    config = load_yaml_mapping(registration_cfg)
    cohort_folder = Path(cohort_folder).expanduser().resolve()
    reference_mouse = Path(reference_mouse).expanduser().resolve()
    config['ref_image_dir'] = str(reference_mouse)
    config['cohort_dir'] = str(cohort_folder)

    try:
        run_snakemake_workflow(script_home, config, cores=9, keep_going=True)
        clean_after_registration(cohort_folder)
    except subprocess.CalledProcessError as e:
        print(f"Error during registration: {e}")
        print("Please check the config file and the cohort folder.")




def load_yaml_mapping(config_path: Path) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping.")
    return data

def _coerce_override_value(value: str, template):
    if isinstance(template, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(template, int):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def apply_config_overrides(config: Dict, overrides: Optional[List[str]]) -> None:
    if not overrides:
        return
    for item in overrides:
        if '=' not in item:
            raise ValueError(f"Invalid override '{item}'. Use KEY=VALUE.")
        key, raw_value = item.split('=', 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            raise ValueError('Override key cannot be empty.')
        if key in config:
            config[key] = _coerce_override_value(value, config[key])
        else:
            config[key] = value


def run_snakemake_workflow(snakefile: Path, config: Dict, cores: int, keep_going: bool) -> None:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.safe_dump(config, temp_file)
        temp_path = temp_file.name
    cmd = [
        'snakemake',
        '--snakefile', str(snakefile),
        '--configfile', temp_path,
        '--cores', str(cores),
    ]
    if keep_going:
        cmd.append('--keep-going')
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
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / 'smk_configs' / 'mouse_skull.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_yaml_mapping(config_path)
    workflows_root = base_dir / 'snakemake_workflows'
    cohort_dir = cohort_dir.expanduser().resolve()
    if not cohort_dir.exists():
        raise FileNotFoundError(f"Cohort directory not found: {cohort_dir}")
    config['cohort_dir'] = str(cohort_dir)

    apply_config_overrides(config, overrides)
    required_keys = {
        'cohort_dir',
        'ref_image_dir',
        'parameter_file_nonlin',
        'parameter_file_lin',
        'mesh_transform_script',
        'slicer_core',
        'ref_im',
        'ref_mask',
        'body_part_detection_top_model',
    }
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(
            'Missing required config keys for full body workflow: ' + ', '.join(sorted(missing))
        )
    run_snakemake_workflow(workflows_root / 'mouse_full_body.smk', config, cores, keep_going)
    run_snakemake_workflow(workflows_root / 'extract_head_from_full_body.smk', config, cores, keep_going)




def compute_2cohort_differences(cohort_1: Dict[str, vtk.vtkPolyData], cohort_2: Dict[str, vtk.vtkPolyData]):
    """
    Compute the statistics for the two cohorts.
    """
    # Compute the statistics for the two cohorts
    print("Computing the statistics for the two cohorts")
    ref_mesh = next(iter(cohort_1.values()))
    n_cells = ref_mesh.GetNumberOfCells()
    def compute_cell_means(mesh: vtk.vtkPolyData) -> np.ndarray:
        means = np.zeros((n_cells, 3))
        for i in range(n_cells):
            cell = ref_mesh.GetCell(i)
            pt_coords = [mesh.GetPoint(cell.GetPointId(j))
                         for j in range(cell.GetNumberOfPoints())]
            means[i] = np.mean(pt_coords, axis=0)
        return means

    cohort_1_means = {key: compute_cell_means(mesh) for key, mesh in cohort_1.items()}
    cohort_2_means = {key: compute_cell_means(mesh) for key, mesh in cohort_2.items()}

    # Compute the differences kolmogorov-smirnov test differences between the two cohorts for each cell

    ks_results = []
    t_test_results = []
    mean_diffs = []
    for i in range(n_cells):
        cells_1 = [cohort_1_means[key][i] for key in cohort_1_means.keys()]
        cells_2 = [cohort_2_means[key][i] for key in cohort_2_means.keys()]
        cells_1 = np.array(cells_1)
        cells_2 = np.array(cells_2)

        p_vals = []

        for j in range(3):
            _, tmp = scipy.stats.ks_2samp(cells_1[:, j], cells_2[:, j], alternative='two-sided')
            p_vals.append(tmp)

        ttest = scipy.stats.ttest_ind(cells_1, cells_2, axis=0)
        mean_diff = np.mean(cells_1, axis=0) - np.mean(cells_2, axis=0)
        print("Mean difference", mean_diff)
        print("ttest", ttest)
        t_test_results.append(ttest)
        mean_diffs.append(mean_diff)
        ks_results.append((None, p_vals))
        #ks_stat, p_value = scipy.stats.ks_2samp(cells_1, cells_2, alternative='two-sided')
        #ks_results.append((ks_stat, p_value))
        # Save the results to a fi

    return {"ks": ks_results, "t_test": t_test_results, "mean_diff": mean_diffs}


def read_vtk_polydata_points_faces(polydata: vtk.vtkPolyData):
    points = np.array([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])
    faces = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        if len(ids) == 3:  # only triangles
            faces.append(ids)
    return points, np.array(faces)

def produce_gif_coh_comparison(ref_mesh: vtk.vtkPolyData,
                               ks_results: list,
                               output_gif_file: str):
    from matplotlib import cm, animation
    from mpl_toolkits.mplot3d import Axes3D

    # Extract points and faces
    print("ref_mesh", ref_mesh)
    pts, faces = read_vtk_polydata_points_faces(ref_mesh)
    print("PTS", pts.shape)
    print("FACES", faces.shape)

    # Color map based on p-value < 0.05
    p_values = np.array([p for _, p in ks_results])

    colors =  [ 0.0 if all(p_values[p,:] > 0.05) else 1.0 for p in range(p_values.shape[0])]
    #
    print( "Ratio of significant p-values", np.sum(colors) / len(colors))
    #colors = np.where(p_values < 0.05, 1.0, 0.0)  # binary coloring
    face_colors = np.array([colors[face[0]] for face in faces])

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('coolwarm')

    norm = plt.Normalize(0, 1)
    trisurf = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                               triangles=faces,
                               linewidth=0.2,
                               antialiased=True)
    from matplotlib.colors import ListedColormap


    cmap = ListedColormap(['lightgray', 'crimson'])  # 0 = gray, 1 = red
    norm = plt.Normalize(0, 1)
    print(len(colors))
    print("colors[0]", colors[0])
    trisurf.set_fc(cmap(norm(colors)))

    ax.set_box_aspect([np.ptp(i) for i in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    #ax.axis('off')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['n.s.', 'p < 0.05'])
    cbar.set_label('Kolmogorov–Smirnov p-value threshold')

    def animate(frame):
        ax.view_init(elev=20, azim=frame * 36)

    ani = animation.FuncAnimation(fig, animate, frames=10)
    ani.save(output_gif_file, writer='pillow', fps=1)
    plt.close(fig)


def produce_gif_ttest_comparison(ref_mesh: vtk.vtkPolyData,
                                    t_test_results: list,
                                    output_gif_file: str):
    from matplotlib import cm, animation
    from mpl_toolkits.mplot3d import Axes3D

        # Extract points and faces
    print("ref_mesh", ref_mesh)
    pts, faces = read_vtk_polydata_points_faces(ref_mesh)
    print("PTS", pts.shape)
    print("FACES", faces.shape)

        # Color map based on p-value < 0.05
    p_values = np.array([p.pvalue for p in t_test_results])

    colors =  [ 0.0 if all(p_values[p,:] > 0.05) else 1.0 for p in range(p_values.shape[0])]
        #
    print( "Ratio of significant p-values", np.sum(colors) / len(colors))
    #colors = np.where(p_values < 0.05, 1.0, 0.0)  # binary coloring
    face_colors = np.array([colors[face[0]] for face in faces])

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('coolwarm')

    norm = plt.Normalize(0, 1)
    trisurf = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                            triangles=faces,
                            linewidth=0.2,
                            antialiased=True)
    from matplotlib.colors import ListedColormap


    cmap = ListedColormap(['lightgray', 'crimson'])  # 0 = gray, 1 = red
    norm = plt.Normalize(0, 1)
    print(len(colors))
    print("colors[0]", colors[0])
    trisurf.set_fc(cmap(norm(colors)))

    ax.set_box_aspect([np.ptp(i) for i in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    #ax.axis('off')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['n.s.', 'p < 0.05'])
    cbar.set_label("T-test p-value threshold")

    def animate(frame):
        ax.view_init(elev=20, azim=frame * 36)

    ani = animation.FuncAnimation(fig, animate, frames=10)
    ani.save(output_gif_file, writer='pillow', fps=1)
    plt.close(fig)


def produce_gif_mean_diff(ref_mesh: vtk.vtkPolyData,
                          mean_diff: list,
                            output_gif_file: str):
    from matplotlib import cm, animation
    from mpl_toolkits.mplot3d import Axes3D

    mean_norms = np.linalg.norm(mean_diff,axis=1)
    print(mean_norms)

    pts, faces = read_vtk_polydata_points_faces(ref_mesh)

    # Назначаем цвет каждой ячейке (по норме)
    face_colors = mean_norms  # one value per face
    p95 = np.percentile(mean_norms, 95)

    # Обрезаем: всё выше 95-го → приравниваем к 95-му
    mean_norms_clipped = np.clip(mean_norms, a_min=None, a_max=p95)
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=face_colors.min(), vmax=face_colors.max())

    # Создаём фигуру и ось
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    trisurf = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                              triangles=faces,
                              linewidth=0.2,
                              antialiased=True)
    trisurf.set_facecolors(cmap(norm(face_colors)))

    # Цветовая шкала
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(mean_norms)  # НЕ пустой массив!
    plt.colorbar(sm, ax=ax)


    # Настройка сцены
    ax.set_box_aspect([np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])])
    #ax.axis('off')

    # Анимация вращения
    def animate(frame):
        ax.view_init(elev=20, azim=frame * 36)

    ani = animation.FuncAnimation(fig, animate, frames=10)
    ani.save(output_gif_file, writer='pillow', fps=1)
    plt.close(fig)


def uproduce_gif_std(ref_mesh: vtk.vtkPolyData,
                    std_results: list,
                    output_gif_file: str,
                     upper_limit: Optional[float] = None):

    from matplotlib import cm, animation
    from mpl_toolkits.mplot3d import Axes3D
    std_norms = np.linalg.norm(std_results,axis=1)
    print(std_norms)
    pts, faces = read_vtk_polydata_points_faces(ref_mesh)
    # Назначаем цвет каждой ячейке (по норме)
    face_colors = std_norms  # one value per face
    p95 = np.percentile(std_norms, 95)
    # Обрезаем: всё выше 95-го → приравниваем к 95-му
    std_norms_clipped = np.clip(std_norms, a_min=None, a_max=p95)
    cmap = plt.get_cmap('viridis')
    if upper_limit is not None:
        norm = plt.Normalize(vmin=0, vmax=upper_limit)
    else:
        norm = plt.Normalize(vmin=0, vmax=face_colors.max())

    # Создаём фигуру и ось

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    trisurf = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                                triangles=faces,
                                linewidth=0.2,
                                antialiased=True)
    trisurf.set_facecolors(cmap(norm(face_colors)))
    # Цветовая шкала
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(std_norms)  # НЕ пустой массив!
    plt.colorbar(sm, ax=ax)
    # Настройка сцены
    ax.set_box_aspect([np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])])
    #ax.axis('off')
    # Анимация вращения
    def animate(frame):
        ax.view_init(elev=20, azim=frame * 36)
    ani = animation.FuncAnimation(fig, animate, frames=10)
    ani.save(output_gif_file, writer='pillow', fps=1)
    plt.close(fig)


def produce_gif_scalar_map(ref_mesh: vtk.vtkPolyData,
                           face_values: np.ndarray,
                           output_gif_file: str,
                           cmap_name: str = 'viridis',
                           clip_percentile: Optional[float] = 95.0,
                           max_value: Optional[float] = None,
                           colorbar_label: str = 'Value'):
    from matplotlib import cm, animation
    # Prepare points and faces
    pts, faces = read_vtk_polydata_points_faces(ref_mesh)
    values = np.asarray(face_values).astype(float)
    if max_value is not None:
        vmax = float(max_value)
        values = np.clip(values, a_min=None, a_max=vmax)
    elif clip_percentile is not None:
        vmax = np.percentile(values, clip_percentile)
        values = np.clip(values, a_min=None, a_max=vmax)
    else:
        vmax = float(np.max(values))

    vmin = float(np.min(values)) if np.isfinite(values).all() else 0.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    trisurf = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                              triangles=faces,
                              linewidth=0.2,
                              antialiased=True)
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    trisurf.set_facecolors(cmap(norm(values)))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(values)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(colorbar_label)
    ax.set_box_aspect([np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])])

    def animate(frame):
        ax.view_init(elev=20, azim=frame * 36)

    from matplotlib import animation as _anim
    ani = _anim.FuncAnimation(fig, animate, frames=10)
    ani.save(output_gif_file, writer='pillow', fps=1)
    plt.close(fig)

def compute_and_save_mean_mesh(cohort: Dict[str, vtk.vtkPolyData], output_file: str):
    """
    Compute the mean mesh for the given cohort and save it to a file.
    """
    # Compute the mean mesh
    gpa = gpa_new(cohort)

    mean_mesh = compute_mean_mesh(gpa)

    # Save the mean mesh to a file
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(mean_mesh)
    writer.Write()

def compute_facewise_centroid_std(meshes: Dict[str, vtk.vtkPolyData]) -> np.ndarray:
    """
    Compute standard deviation of face centroids across meshes.

    Returns:
        face_std: np.ndarray of shape (n_faces,), where each value is the std of centroid positions for a face.
    """
    mesh_list = list(meshes.values())
    n_meshes = len(mesh_list)
    ref_mesh = mesh_list[0]
    n_faces = ref_mesh.GetNumberOfCells()

    # (n_faces, n_meshes, 3): центроиды каждой ячейки в каждом меше
    face_centroids = np.zeros((n_faces, n_meshes, 3), dtype=np.float32)

    for m_idx, mesh in enumerate(mesh_list):
        for f_idx in range(n_faces):
            cell = mesh.GetCell(f_idx)
            pts = [np.array(mesh.GetPoint(cell.GetPointId(i))) for i in range(cell.GetNumberOfPoints())]
            centroid = np.mean(pts, axis=0)
            face_centroids[f_idx, m_idx] = centroid
    print("fcs", face_centroids.shape)
    # std по каждой координате, потом норма → shape: (n_faces,)
    face_std = np.std(face_centroids, axis=1)  # → (n_faces, 3)
    #face_std_norm = np.linalg.norm(face_std, axis=1)  # → (n_faces,)

    return face_std



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


def _make_polydata_actor(polydata: vtk.vtkPolyData,
                         color: Tuple[float, float, float],
                         opacity: float,
                         wireframe: bool = False) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(opacity)
    if wireframe:
        prop.SetRepresentationToWireframe()
        prop.SetLineWidth(1.5)
    else:
        prop.SetRepresentationToSurface()
    return actor


def launch_procrustes_viewer(subject_name: str,
                             subject_mesh: vtk.vtkPolyData,
                             mean_mesh: Optional[vtk.vtkPolyData],
                             peer_meshes: Dict[str, vtk.vtkPolyData]) -> None:
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.08, 0.08, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName(f"Procrustes viewer - {subject_name}")
    render_window.SetSize(1200, 900)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    subject_actor = _make_polydata_actor(subject_mesh, (0.88, 0.36, 0.30), 0.9)
    renderer.AddActor(subject_actor)

    mean_actor = None
    if mean_mesh is not None:
        mean_actor = _make_polydata_actor(mean_mesh, (0.25, 0.65, 1.0), 0.35)
        renderer.AddActor(mean_actor)

    peer_order = list(peer_meshes.keys())
    peer_actors = []
    for name in peer_order:
        actor = _make_polydata_actor(peer_meshes[name], (0.25, 0.9, 0.55), 0.3)
        actor.VisibilityOff()
        renderer.AddActor(actor)
        peer_actors.append(actor)

    text_actor = vtk.vtkTextActor()
    text_actor.SetPosition(10, 10)
    text_prop = text_actor.GetTextProperty()
    text_prop.SetFontSize(14)
    text_prop.SetColor(1.0, 1.0, 1.0)
    text_prop.SetFontFamilyToArial()
    renderer.AddActor2D(text_actor)

    state = {
        "subject_visible": True,
        "mean_visible": mean_actor is not None,
        "peer_visible": False,
        "peer_index": 0,
        "wireframe": False,
    }

    def update_peer_visibility():
        if not peer_actors:
            return
        for idx, actor in enumerate(peer_actors):
            actor.SetVisibility(state["peer_visible"] and idx == state["peer_index"])

    def rebuild_text():
        lines = [
            f"Subject: {subject_name} ({'visible' if state['subject_visible'] else 'hidden'})"
        ]
        if mean_actor is not None:
            lines.append(f"Mean: {'visible' if state['mean_visible'] else 'hidden'}")
        if peer_order:
            if state["peer_visible"]:
                current_peer = peer_order[state["peer_index"]]
                lines.append(f"Peer: {current_peer}")
            else:
                lines.append("Peer: hidden")
        lines.append("")
        lines.append("Keys: s toggle subject, m toggle mean, space toggle peer, n/p cycle peer, w wireframe, q quit")
        text_actor.SetInput("\n".join(lines))

    def set_subject_wireframe(enabled: bool):
        prop = subject_actor.GetProperty()
        if enabled:
            prop.SetRepresentationToWireframe()
            prop.SetLineWidth(1.5)
        else:
            prop.SetRepresentationToSurface()
            prop.SetLineWidth(1.0)

    def on_key_press(obj, _event):
        key = obj.GetKeySym()
        if key == "s":
            state["subject_visible"] = not state["subject_visible"]
            subject_actor.SetVisibility(state["subject_visible"])
        elif key == "m" and mean_actor is not None:
            state["mean_visible"] = not state["mean_visible"]
            mean_actor.SetVisibility(state["mean_visible"])
        elif key in ("space", "v"):
            if peer_actors:
                state["peer_visible"] = not state["peer_visible"]
                update_peer_visibility()
        elif key in ("n", "Right"):
            if peer_actors:
                state["peer_index"] = (state["peer_index"] + 1) % len(peer_actors)
                state["peer_visible"] = True
                update_peer_visibility()
        elif key in ("p", "Left"):
            if peer_actors:
                state["peer_index"] = (state["peer_index"] - 1) % len(peer_actors)
                state["peer_visible"] = True
                update_peer_visibility()
        elif key == "w":
            state["wireframe"] = not state["wireframe"]
            set_subject_wireframe(state["wireframe"])
        elif key in ("q", "Escape"):
            render_window.Finalize()
            obj.TerminateApp()
            return
        else:
            return
        rebuild_text()
        render_window.Render()

    interactor.AddObserver("KeyPressEvent", on_key_press)
    rebuild_text()
    set_subject_wireframe(False)
    renderer.ResetCamera()
    render_window.Render()
    interactor.Initialize()
    interactor.Start()


def visualize_subject_interactive(subject_path: Path,
                                  reference_cohort: Path) -> None:
    """
    Launch an interactive VTK viewer for a single subject against a reference cohort.

    subject_path:
        Path to a subject directory or directly to its head_affine_registered.obj file.
    reference_cohort:
        Path to a cohort directory that already contains registered meshes.
    """
    subject_path = Path(subject_path)
    reference_cohort = Path(reference_cohort)
    if not reference_cohort.exists():
        raise FileNotFoundError(f"Reference cohort directory not found: {reference_cohort}")

    if subject_path.is_file():
        if subject_path.name != 'head_affine_registered.obj':
            raise ValueError('Subject file must be head_affine_registered.obj if a file path is provided.')
        subject_dir = subject_path.parent
    else:
        subject_dir = subject_path
        subject_path = subject_dir / 'head_affine_registered.obj'

    if not subject_path.exists():
        raise FileNotFoundError(f"Subject mesh not found: {subject_path}")

    subject_name = subject_dir.name

    ref_meshes = collect_registered_meshes(reference_cohort)

    gpa_input: "OrderedDict[str, Path]" = OrderedDict()
    for name, mesh_path in ref_meshes.items():
        gpa_input[f"ref::{name}"] = mesh_path
    subject_key = f"query::{subject_name}"
    gpa_input[subject_key] = subject_path

    print(f"Running Procrustes alignment against {len(ref_meshes)} reference meshes...")
    aligned_meshes = gpa_new(gpa_input, modes='rigid')

    print("Finished alignment. Collecting meshes for visualization...")

    aligned_subject = aligned_meshes[subject_key]
    aligned_reference = OrderedDict(
        (key, aligned_meshes[key]) for key in aligned_meshes.keys() if key.startswith('ref::')
    )
    print(f"Aligned subject: {aligned_subject.GetNumberOfPoints()} points, {aligned_subject.GetNumberOfCells()} cells")

    
    mean_mesh = compute_mean_mesh(aligned_reference) if aligned_reference else None
    print("Computed mean mesh.")
    peer_meshes: "OrderedDict[str, vtk.vtkPolyData]" = OrderedDict(
        (key.split('::', 1)[1], aligned_reference[key]) for key in aligned_reference.keys()
    )

    print("Launching interactive viewer. Close the window or press 'q' to exit.")
    launch_procrustes_viewer(subject_name, aligned_subject, mean_mesh, peer_meshes)



def should_recompute(area_percent: float, percent_points_over: float) -> bool:
    return area_percent >= 40.0 and percent_points_over >= 40.0


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

    extract_smk = Path(__file__).resolve().parent / "snakemake_workflows" / "extract_head_from_full_body.smk"
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

    config = load_yaml_mapping(Path(__file__).resolve().parent / "smk_configs" / "mouse_skull.yaml")
    config["ref_image_dir"] = str(reference_mouse)
    config["cohort_dir"] = str(cohort_dir)
    config["subjects"] = subjects
    reg_smk = Path(__file__).resolve().parent / "snakemake_workflows" / "generic_coregistration.smk"
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
            for row in reader:
                subject = row["subject"]
                area_val = float(row["area_percent_mahal_gt_5"])
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
                        subject_info: Dict[str, Dict]) -> Tuple[set, bool]:
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
                if should_recompute(area_percent, percent_over):
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
            "area_percent_mahal_gt_5": info["area_percent"],
        })
        for face_idx, mahal_val in enumerate(info["mahal"]):
            mahal_rows.append({
                "subject": key,
                "face_index": face_idx,
                "mahalanobis": float(mahal_val),
            })


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
            area_latest[row["subject"]] = row["area_percent_mahal_gt_5"]
        area_rows[:] = [
            {"subject": subject, "area_percent_mahal_gt_5": area_val}
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
            writer = csv.DictWriter(fh, fieldnames=["subject", "area_percent_mahal_gt_5"])
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
                                fixate_plotting_max: Optional[float] = None):
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

    area_threshold = 5.0
    area_rows = []
    mahal_rows = []
    area_csv = res_dir_new / "mahalanobis_area_gt_5.csv"
    mahal_csv = res_dir_new / "mahalanobis_per_face.csv"
    cached_area, cached_mahal, write_cache = _load_mahalanobis_cache(area_csv, mahal_csv)
    subject_info, computed_cache = _compute_subject_info(
        query_cohort_meshes, ref_cohort_meshes, cached_area, cached_mahal, area_threshold
    )
    write_cache = write_cache or computed_cache

    # Recompute outliers and flag them for GIF regeneration.
    outliers = [k for k, info in subject_info.items()
                if should_recompute(info["area_percent"], info["percent_over"])]
    _remove_outlier_gifs(outliers, res_dir_new, ref_distribution)
    force_gif_subjects, recomputed_cache = _recompute_outliers(
        outliers,
        query_cohort_meshes,
        ref_cohort_meshes,
        query_cohort_dir,
        reference_mouse,
        area_threshold,
        subject_info,
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





def one_cohort_statistics(cohort: str, reference_mouse: str, result_dir: str,upper_limit: Optional[float] = None):
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
    coh_aligned = gpa_new(cohort_meshes, modes="rigid")




    std_results = compute_facewise_centroid_std(coh_aligned)

    print("std_results", std_results)

    mesh_file = next(iter(coh_aligned.values()))
    # plot results

    uproduce_gif_std(mesh_file,std_results, result_dir / f"{cohort.name}_std.gif", upper_limit=upper_limit)



def main():
    print(f"[INFO] mouse_skull_pipelines version={_get_app_version()}")
    parser = argparse.ArgumentParser(description="Run the full body and skull pipeline.")
    sbp = parser.add_subparsers(dest="cmd", required=True)

    cmp_cohorts = sbp.add_parser("compare_cohorts", help="Compare cohorts at group or per-subject level.")
    cmp_cohorts.add_argument(
        "--cohort_1",
        type=str,
        required=True,
        help="Path to the cohort directory.",
    )
    cmp_cohorts.add_argument(
        "--cohort_2",
        type=str,
        required=True,
        help="Path to the cohort directory.",
    )
    cmp_cohorts.add_argument(
        "--reference_mouse",
        type=str,
        required=True,
        help="Path to the reference mouse directory.",
    )


    cmp_cohorts.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to the result file.",
    )

    cmp_cohorts.add_argument(
        "--method",
        type=str,
        choices=["group", "per_subject"],
        default="group",
        help="Comparison method: group (default) or per_subject",
    )

    cmp_cohorts.add_argument(
        "--ref_distribution",
        type=str,
        choices=["cohort_1", "cohort_2"],
        default="cohort_2",
        help="For per_subject: which cohort defines the reference distribution.",
    )
    cmp_cohorts.add_argument(
        "--fixate_plotting_max",
        type=float,
        default=None,
        help="For per_subject: cap Mahalanobis distances and colorbar at this value.",
    )

    full_body_parser = sbp.add_parser("generate_full_body", help="Generate full body volumes and extract head CTs.")
    full_body_parser.add_argument("--cohort-dir", type=str, required=True, help="Path to the cohort directory containing raw full body scans.")
    full_body_parser.add_argument("--ref-image-dir", dest="ref_image_dir", type=str, default=None, help="Override the reference image directory in the config.")
    full_body_parser.add_argument("--cores", type=int, default=8, help="Number of CPU cores to pass to Snakemake.")
    full_body_parser.add_argument("--keep-going", action="store_true", help="Pass --keep-going to Snakemake.")
    full_body_parser.add_argument("--config-override", action="append", metavar="KEY=VALUE", default=[], help="Override additional config entries (repeatable).")

    viz_parser = sbp.add_parser("visualize_subject", help="Inspect a subject against a reference cohort interactively.")
    viz_parser.add_argument("subject", type=str, help="Path to a subject directory or head_affine_registered.obj file.")
    viz_parser.add_argument("reference_cohort", type=str, help="Path to the reference cohort directory.")

    heatmap_parser = sbp.add_parser(
        "mahalanobis_csv_heatmap",
        help="Create a Mahalanobis heatmap GIF from per-face CSV and a reference mesh.",
    )
    heatmap_parser.add_argument("--mahal_csv", type=str, required=True, help="Path to CSV with subject/subj_id, face_index/face_id, mahalanobis.")
    heatmap_parser.add_argument("--reference_mesh", type=str, required=True, help="Path to reference OBJ mesh.")
    heatmap_parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated GIF(s).")
    heatmap_parser.add_argument("--subject", type=str, default=None, help="Subject name to render. If omitted, render all subjects in CSV.")
    heatmap_parser.add_argument("--fixate_plotting_max", type=float, default=None, help="Optional fixed max value for color scale.")

    self_aver_parser = sbp.add_parser("one_cohort_statistics", help="Run the cohort statistics explanation.")

    self_aver_parser.add_argument(
        "--cohort",
        type=str,
        required=True,
        help="Path to the cohort directory.",
    )
    self_aver_parser.add_argument(
        "--reference_mouse",
        type=str,
        required=True,
        help="Path to the reference mouse directory.",
    )
    self_aver_parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to the result folder.",
    )
    self_aver_parser.add_argument(
        "--upper_limit",
        type=float,
        default=None,
        required=False,
        help="Path to the result file.",
    )

    self_aver_parser.add_argument("--save_statics",
                                      action="store_true",
                                      help="Save the statistics to a file.")



    args = parser.parse_args()
    print(args)


    # Check if the result directory exists, if not create it
    if args.cmd == "compare_cohorts":
        result_dir = Path(args.result_dir)
        if not result_dir.exists():
            os.makedirs(result_dir)
        # Branch by method
        if getattr(args, "method", "group") == "per_subject":
            compare_cohorts_per_subject(
                args.cohort_1,
                args.cohort_2,
                args.reference_mouse,
                result_dir,
                ref_distribution=getattr(args, "ref_distribution", "cohort_2"),
                fixate_plotting_max=getattr(args, "fixate_plotting_max", None),
            )
        else:
            # Default group-level comparisons and visualizations
            compare_cohorts(args.cohort_1, args.cohort_2, args.reference_mouse, result_dir)

    elif args.cmd == "visualize_subject":
        visualize_subject_interactive(
            subject_path=Path(args.subject),
            reference_cohort=Path(args.reference_cohort),
        )

    elif args.cmd == "one_cohort_statistics":
        result_dir = Path(args.result_dir)
        if not result_dir.exists():
            os.makedirs(result_dir)
        one_cohort_statistics(args.cohort, args.reference_mouse, result_dir, upper_limit=args.upper_limit)

    elif args.cmd == "mahalanobis_csv_heatmap":
        create_mahalanobis_heatmap_from_csv(
            mahal_csv=Path(args.mahal_csv),
            reference_mesh=Path(args.reference_mesh),
            output_dir=Path(args.output_dir),
            subject=getattr(args, "subject", None),
            fixate_plotting_max=getattr(args, "fixate_plotting_max", None),
        )

    elif args.cmd == "generate_full_body":
        prepare_full_body_and_extract_heads(
            cohort_dir=Path(args.cohort_dir),
            ref_image_dir=Path(args.ref_image_dir) if args.ref_image_dir else None,
            overrides=args.config_override,
            cores=9,
            keep_going=True,
        )


if __name__ == "__main__":
    main()




