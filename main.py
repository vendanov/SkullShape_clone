
import argparse
import os
import shutil
import subprocess
import sys
from typing import Dict, Optional
import scipy.io as scio
import numpy as np
import vtk
import yaml

import tempfile

from pathlib import Path

from matplotlib import pyplot as plt
from sympy.physics.quantum.identitysearch import scipy

from mouse_skull_pipelines.scripts.gpa import gpa_new, compute_mean_mesh

def clean_after_registration(cohort_folder: Path):
    """
    Clean the cohort folder after registration.
    """

    files_to_leave = ["head_affine_registered.obj", "4_inv_mesh.flag","ct_image.nii.gz",
                      "nl_transformation_params.txt" , "affine_head.txt"
                      ]

    # itearate over the files in the cohort folder and subfolders
    for root, dirs, files in os.walk(cohort_folder):
        for file in files:
            # if the file is not in the list of files to leave, remove it
            if file not in files_to_leave:
                file_path = os.path.join(root, file)
                print(f"Removing {file_path}")
                os.remove(file_path)

def registration_to_reference(cohort_folder: Path, reference_mouse):

    """
    Check if the registration is done for the given cohort folder.
    """

    # read config file
    # check if the registration is done

    # current file path
    current_file = Path(__file__).resolve()
    parent_dir = current_file.parent
    registration_cfg = parent_dir / 'smk_configs' / 'mouse_skull.yaml'

    script_home = parent_dir / 'snakemake_workflows' / 'generic_coregistration.smk'

    # read yaml file

    with open(registration_cfg, 'r') as file:
        config = yaml.safe_load(file)
        config['ref_image_dir'] = str(reference_mouse)
        config['cohort_dir'] = str(cohort_folder)

        # save the temporary config file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as temp_file:

            try:
                temp_file.write(yaml.dump(config).encode())
                temp_file.close()
                temp_file_path = temp_file.name
                print(f"Temporary config file created at: {temp_file_path}")
                # run the temporary config file registration
                # run the registration
                snakemake_command = f"snakemake --snakefile {script_home} --configfile {temp_file_path} --cores 9  --keep-going"
                cmd = snakemake_command.split()
                subprocess.run(cmd, check=True)
                #clean_after_registration(cohort_folder)
            except Exception as e:
                print(f"Error during registration: {e}")
                print("Please check the config file and the cohort folder.")
            finally:
                # remove the temporary config file
                temp_file.close()
                os.remove(temp_file.name)

            # remove the temporary config file
            #os.remove(temp_file_path)



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
    cmap = cm.get_cmap('coolwarm')

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
    cmap = cm.get_cmap('coolwarm')

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
    cmap = cm.get_cmap('viridis')
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
    cmap = cm.get_cmap('viridis')
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
    cmap = cm.get_cmap(cmap_name)
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

    # Collect meshes
    cohort_1_meshes = {x.name: x / "head_affine_registered.obj" for x in Path(cohort_1).iterdir()
                       if (x / "head_affine_registered.obj").exists()}
    cohort_2_meshes = {x.name: x / "head_affine_registered.obj" for x in Path(cohort_2).iterdir()
                       if (x / "head_affine_registered.obj").exists()}

    print(f"Found meshes -> cohort_1: {len(cohort_1_meshes)}, cohort_2: {len(cohort_2_meshes)}")

    # select reference cohort for distribution here
    if ref_distribution == "cohort_1":
        ref_cohort_meshes = cohort_1_meshes
        query_cohort_meshes = cohort_2_meshes

        res_dir_new = result_dir / cohort_2.name
        
    else:
        ref_cohort_meshes = cohort_2_meshes
        query_cohort_meshes = cohort_1_meshes
        res_dir_new = result_dir / cohort_1.name
    res_dir_new.mkdir(parents=True, exist_ok=True)

    # now per each mouse in query, compute GPA with reference cohort

    for key, mesh_path in query_cohort_meshes.items():
        print(f"Loading query mesh {key} from {mesh_path}")
        
        gpa_all = gpa_new({**ref_cohort_meshes, key: mesh_path}, modes="rigid")

        # Build reference cohort (aligned) and the aligned key mesh
        aligned_key_mesh: vtk.vtkPolyData = gpa_all[key]
        aligned_ref_meshes = [gpa_all[k] for k in gpa_all.keys() if k != key]

        # Extract faces once (assume identical topology across aligned meshes)
        pts_key, faces = read_vtk_polydata_points_faces(aligned_key_mesh)
        n_faces = faces.shape[0]

        # Use per-point empirical covariance (not face centroids)
        from sklearn.covariance import EmpiricalCovariance
        # Stack reference meshes' points: (n_ref, n_points, 3)
        ref_points_stack = np.stack([read_vtk_polydata_points_faces(m)[0] for m in aligned_ref_meshes], axis=0)
        n_ref, n_points, _ = ref_points_stack.shape

        # Fit EmpiricalCovariance per point with light trimming (99th percentile)
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
            # Fallback if insufficient refs after trimming
            mu_p = X.mean(axis=0)
            C = np.cov(X, rowvar=False) if n_ref > 1 else np.eye(3) * eps
            C = np.asarray(C, dtype=np.float64)
            C.flat[::4] += eps
            mu_pts[p] = mu_p
            prec_pts[p] = np.linalg.pinv(C)

        # Per-point Mahalanobis for the query subject
        deltas = pts_key - mu_pts  # (n_points, 3)
        d2_pts = np.einsum('pi,pij,pj->p', deltas, prec_pts, deltas, optimize=True)
        d2_pts = np.clip(d2_pts, 0.0, None)

        # Aggregate per face as RMS over the face's vertices
        mahal = np.sqrt(d2_pts[faces].mean(axis=1))

        # Produce rotating GIF of Mahalanobis distances
        out_path = res_dir_new / f"{key}_to_{ref_distribution}_mahalanobis.gif"
        produce_gif_scalar_map(aligned_key_mesh, mahal, str(out_path),
                               cmap_name='magma',
                               clip_percentile=None if fixate_plotting_max is not None else 95.0,
                               max_value=fixate_plotting_max,
                               colorbar_label='Mahalanobis distance')


        print(f"Saved Mahalanobis distance GIF for {key} at {out_path}")


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

    if args.cmd == "one_cohort_statistics":
        result_dir = Path(args.result_dir)
        if not result_dir.exists():
            os.makedirs(result_dir)
        one_cohort_statistics(args.cohort, args.reference_mouse, result_dir,upper_limit=args.upper_limit)


if __name__ == "__main__":
    main()




