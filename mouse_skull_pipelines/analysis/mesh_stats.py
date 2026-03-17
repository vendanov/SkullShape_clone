from typing import Dict

import numpy as np
import scipy
import vtk

from mouse_skull_pipelines.scripts.gpa import compute_mean_mesh, gpa_new


def compute_2cohort_differences(cohort_1: Dict[str, vtk.vtkPolyData], cohort_2: Dict[str, vtk.vtkPolyData]):
    print("Computing the statistics for the two cohorts")
    ref_mesh = next(iter(cohort_1.values()))
    n_cells = ref_mesh.GetNumberOfCells()

    def compute_cell_means(mesh: vtk.vtkPolyData) -> np.ndarray:
        means = np.zeros((n_cells, 3))
        for i in range(n_cells):
            cell = ref_mesh.GetCell(i)
            pt_coords = [mesh.GetPoint(cell.GetPointId(j)) for j in range(cell.GetNumberOfPoints())]
            means[i] = np.mean(pt_coords, axis=0)
        return means

    cohort_1_means = {key: compute_cell_means(mesh) for key, mesh in cohort_1.items()}
    cohort_2_means = {key: compute_cell_means(mesh) for key, mesh in cohort_2.items()}

    ks_results = []
    t_test_results = []
    mean_diffs = []
    for i in range(n_cells):
        cells_1 = np.array([cohort_1_means[key][i] for key in cohort_1_means.keys()])
        cells_2 = np.array([cohort_2_means[key][i] for key in cohort_2_means.keys()])

        p_vals = []
        for j in range(3):
            _, tmp = scipy.stats.ks_2samp(cells_1[:, j], cells_2[:, j], alternative="two-sided")
            p_vals.append(tmp)

        ttest = scipy.stats.ttest_ind(cells_1, cells_2, axis=0)
        mean_diff = np.mean(cells_1, axis=0) - np.mean(cells_2, axis=0)
        print("Mean difference", mean_diff)
        print("ttest", ttest)
        t_test_results.append(ttest)
        mean_diffs.append(mean_diff)
        ks_results.append((None, p_vals))

    return {"ks": ks_results, "t_test": t_test_results, "mean_diff": mean_diffs}


def read_vtk_polydata_points_faces(polydata: vtk.vtkPolyData):
    points = np.array([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])
    faces = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        if len(ids) == 3:
            faces.append(ids)
    return points, np.array(faces)


def compute_and_save_mean_mesh(cohort: Dict[str, vtk.vtkPolyData], output_file: str):
    # Use rigid GPA to preserve physical scale (similarity mode rescales meshes).
    gpa = gpa_new(cohort, modes="rigid")
    mean_mesh = compute_mean_mesh(gpa)
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(mean_mesh)
    writer.Write()


def compute_facewise_centroid_std(meshes: Dict[str, vtk.vtkPolyData]) -> np.ndarray:
    mesh_list = list(meshes.values())
    n_meshes = len(mesh_list)
    ref_mesh = mesh_list[0]
    n_faces = ref_mesh.GetNumberOfCells()

    face_centroids = np.zeros((n_faces, n_meshes, 3), dtype=np.float32)
    for m_idx, mesh in enumerate(mesh_list):
        for f_idx in range(n_faces):
            cell = mesh.GetCell(f_idx)
            pts = [np.array(mesh.GetPoint(cell.GetPointId(i))) for i in range(cell.GetNumberOfPoints())]
            centroid = np.mean(pts, axis=0)
            face_centroids[f_idx, m_idx] = centroid
    print("fcs", face_centroids.shape)
    return np.std(face_centroids, axis=1)
