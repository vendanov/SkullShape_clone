import pickle
from pathlib import Path
from typing import Dict
import numpy as np
from tqdm.auto import tqdm
#from concurrent.futures import ProcessPoolExecutor
from procrustes import generic
import vtk

from mouse_skull_pipelines.scripts.mesh_algorithms import read_mesh_from_file


def procreustes_alignment(reference_mesh, compute_mesh):
    #("STARTED")
    if not isinstance(reference_mesh,np.ndarray):
        reference_mesh = np.array(reference_mesh)
    if not isinstance(compute_mesh,np.ndarray):
        compute_mesh = np.array(compute_mesh)

    procr_result = generic(a=compute_mesh,b=reference_mesh)
    return procr_result.new_a, procr_result.error, procr_result.t


def gpa(meshes: Dict[str, np.ndarray], max_iterations=100, threshold=1e-14):
    num_meshes = len(meshes)
    num_vertices = len(list(meshes.values())[0])

    # Convert meshes to 3D numpy array
    stacked_meshes = np.stack(list(meshes.values()))

    mesh_keys = list(meshes.keys())

    # Initialize reference shape as the first mesh
    reference_mesh = stacked_meshes[0]

    finishing_iteration = 0 #
    for iteration in tqdm(range(max_iterations)):

        # Step 1: Superimpose all meshes to the current reference shape
        aligned_meshes = [reference_mesh]

        # futures = [executor.submit(procreustes_alignment, reference_mesh, mesh) for mesh in stacked_meshes[1:]]
        #
        # for f in tqdm(futures):
        #     aligned_meshes.append( f.result() )

        for mesh in tqdm(stacked_meshes[1:]):
            tmp,_,_= procreustes_alignment(reference_mesh, mesh)
            aligned_meshes.append(tmp)
        # Step 2: Compute the mean shape of the current set of superimposed shapes
        mean_mesh = np.mean(np.stack(aligned_meshes), axis=0)
        # print("mean mesh shape", mean_mesh.shape)
        # Step 3: Check for convergence based on Procrustes distance
        _,procrustes_distance,_= procreustes_alignment(reference_mesh, mean_mesh)
        print("Procrustes distance ", procrustes_distance)
        if procrustes_distance < threshold:
            finishing_iteration = iteration
            break

        # Update the reference shape
        reference_mesh = mean_mesh

    result_shapes = {}
    for key, shape in tqdm(meshes.items()):
        #print(shape.shape)
        result,_,_= procreustes_alignment(reference_mesh=reference_mesh, compute_mesh=shape)
        result_shapes[key] = result

    return reference_mesh, aligned_meshes, result_shapes


def gpa_new(meshes: Dict[str, Path],modes: str = "similarity"):
    """
    Perform Generalized Procrustes Analysis (GPA) on a set of meshes.
    Args:
        meshes: A dictionary where keys are mesh names and values are paths to mesh files.
    Returns:

    """
    # reading meshes
    keys = list(meshes.keys())

    input_list = vtk.vtkMultiBlockDataSet()
    print(meshes.keys())
    for i, f in enumerate(keys):
        pd = read_mesh_from_file(meshes[f])
        input_list.SetBlock(i, pd)
    procrustes = vtk.vtkProcrustesAlignmentFilter()

    procrustes.SetInputData(input_list)
    if modes == "similarity":
        procrustes.GetLandmarkTransform().SetModeToSimilarity()
    elif modes == "rigid":
        procrustes.GetLandmarkTransform().SetModeToRigidBody()
    elif modes == "affine":
        procrustes.GetLandmarkTransform().SetModeToAffine()
    else:
        raise ValueError("Unknown mode. Use 'similarity', 'rigid', or 'affine'.")
    procrustes.Update()

    aligned_output = procrustes.GetOutput()
    print(list(enumerate(meshes.keys())))
    aligned_meshes = {
        key: (aligned_output.GetBlock(i) )for i, key in enumerate(meshes.keys())
    }
    return aligned_meshes


def compute_mean_mesh( meshes: Dict[str, vtk.vtkPolyData]):
    """
    Compute the mean mesh from a list of meshes.
    Args:
        meshes: A dictionary where keys are mesh names and values are vtkPolyData objects.
    Returns:
        A vtkPolyData object representing the mean mesh.
    """
    mesh_list = list(meshes.values())
    n_points = mesh_list[0].GetNumberOfPoints()
    n_meshes = len(mesh_list)

    # Суммируем координаты всех точек
    sum_points = np.zeros((n_points, 3), dtype=np.float64)
    for mesh in mesh_list:
        for i in range(n_points):
            sum_points[i] += np.array(mesh.GetPoint(i))

    mean_points = sum_points / n_meshes

    # Создаём новый PolyData с теми же ячейками, но средними точками
    mean_polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    for pt in mean_points:
        vtk_points.InsertNextPoint(pt.tolist())
    mean_polydata.SetPoints(vtk_points)
    mean_polydata.SetPolys(mesh_list[0].GetPolys())  # копируем faces из первого меша
    return mean_polydata








def read_file(filename):
    f = open(filename,'rb')
    res =pickle.load(f)
    f.close()
    return res

def write_obj_2_file(obj,filename):
    f = open(filename,"wb")
    pickle.dump(obj,f)
    f.close()
if __name__ == "__main__" :
    inp = r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/meshes_dict_snapshot.pkl"
    out = r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/after_procrustes.pkl"
    meshes = read_file(inp)
    res = gpa(meshes)

    write_obj_2_file(res,out)

