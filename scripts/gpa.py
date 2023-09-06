import pickle
from typing import Dict

import scipy.spatial as sp
import numpy as np
from tqdm.auto import tqdm
#from concurrent.futures import ProcessPoolExecutor



def procreustes_alignment(reference_mesh, compute_mesh):
    #("STARTED")
    matx1,mesh_aligned, translation = sp.procrustes(reference_mesh, compute_mesh)

    return mesh_aligned


def gpa(meshes: Dict[str, np.ndarray], max_iterations=4, threshold=1e-6):
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
            aligned_meshes.append(procreustes_alignment(reference_mesh, mesh))
        # Step 2: Compute the mean shape of the current set of superimposed shapes
        mean_mesh = np.mean(np.stack(aligned_meshes), axis=0)
        # print("mean mesh shape", mean_mesh.shape)
        # Step 3: Check for convergence based on Procrustes distance
        _, _, procrustes_distance = sp.procrustes(reference_mesh, mean_mesh)
        print("procrustes distance ", procrustes_distance)
        if procrustes_distance < threshold:
            finishing_iteration = iteration
            break

        # Update the reference shape
        reference_mesh = mean_mesh

    result_shapes = {}
    for key, shape in tqdm(meshes.items()):
        #print(shape.shape)
        result = procreustes_alignment(reference_mesh=reference_mesh, compute_mesh=shape)
        result_shapes[key] = result

    return reference_mesh, aligned_meshes, result_shapes



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

