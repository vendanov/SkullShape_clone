import copy
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import List, Dict
import vtk
import numpy as np

import scipy.io as scio
from tqdm.auto import tqdm
import scipy.spatial as sp
import sys
sys.path.append('..' + os.sep + 'scripts')
from gpa import gpa
from vtk_helper import center_of_mesh
PROCESS_DIR = Path(config['process_dir'])
print(PROCESS_DIR)
#pathlib.Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir')
REF_IMAGE = Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/reference_image/')

ref_im = REF_IMAGE / 'ct_image_.nii.gz'
ref_mask = REF_IMAGE / 'bone_coregistration_mask.nii.gz'


MALE_FACE_PERC = r"\\primus.img.cas.cz\data\83_BIOINFORMATICS\Igor\workdir\wt_male_f_percentiles.mat"
FEMALE_FACE_PERC = r"\\primus.img.cas.cz\data\83_BIOINFORMATICS\Igor\workdir\wt_female_f_percentiles.mat"
coh_file  = r"C:\Users\Public\wt_cohorts_pts.pkl"
coh_file_l = open(coh_file,'rb')
COHORTS = pickle.load(coh_file_l)
coh_file_l.close()
print("FILE READ")

MICE_FOLDERS = [x for x in PROCESS_DIR.iterdir() if (x.is_dir() #and "Trab" in x.name
                and (x/ "head_affine_registered.obj").exists())]
print("START BUILD")
from sklearn.covariance import EmpiricalCovariance,MinCovDet


def procreustes_alignment(reference_mesh, compute_mesh):
    #("STARTED")
    matx1, mesh_aligned, translation = sp.procrustes(reference_mesh,compute_mesh)

    return mesh_aligned





def single_to_many(mesh_pts: np.ndarray, cohort: List[np.ndarray]):
    # build distributions
    num_of_points = mesh_pts.shape[0]
    dists = []
    for i in tqdm(range(num_of_points)):  # i in range of num_of_points
        points = []
        #print (i, working_dict)

        for mesh_v in cohort:
            points.append(mesh_v[i])

        points = np.stack(points)

        emp_cov = EmpiricalCovariance()
        emp_cov.fit(points)
        mahalanobis_distances = emp_cov.mahalanobis(points)
        points = points[np.where(mahalanobis_distances < np.percentile(mahalanobis_distances,99))[0]] # smart thresholding?
        emp_cov.fit(points)
        mahalanobis_distance = emp_cov.mahalanobis([mesh_pts[i]])[0]
        dists.append(mahalanobis_distance)

    return dists

def read_mesh_and_points ( obj_filename ):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_filename)
    reader.Update()
    mesh = reader.GetOutput()
    points = mesh.GetPoints()

    res_mesh = []
    for i in range(points.GetNumberOfPoints()):
        pt = points.GetPoint(i)
        res_mesh.append(pt)

    return np.array(res_mesh), mesh

rule all:
    input:
        expand("{dirname}/mahalanobis.gif",dirname=MICE_FOLDERS),
        expand("{dirname}/percentile.gif", dirname=MICE_FOLDERS)

rule compute_mahalanobis_heads:
    input: "{dirname}/head_affine_registered.obj"
    output: "{dirname}/mahalanobis.mat"
    threads: 1
    run:
        par_name = Path(input[0]).parent.name
        cohort = copy.copy(COHORTS[par_name.split("_")[1]]) # copy males or females

        mesh_pts, mesh = read_mesh_and_points(input[0])


        com = center_of_mesh(mesh)
        cohort[par_name] = mesh_pts - np.array(com)

        #test_V = np.stack(cohort.values())
        _,_,shapes = gpa(cohort ,max_iterations=100, threshold=1e-11)

        art = shapes[par_name]
        shapes.pop(par_name)
        dists = single_to_many(mesh_pts=art,cohort=list(shapes.values()))

        # analyse distances

        polygons = mesh.GetPolys()
        faces = []
        # Iterate over the polygons to extract the faces
        polygons.InitTraversal()
        while True:
            faceIds = vtk.vtkIdList()
            if polygons.GetNextCell(faceIds):
                face = [faceIds.GetId(i) for i in range(faceIds.GetNumberOfIds())]
                faces.append(face)
            else:
                break

        ### compute colors and write to mat
        energies = []
        for x in faces:
            en = 0
            for p in x:
                en += dists[p]
            en = en / 3
            energies.append(en)
        scio.savemat(output[0],{"energies": energies})


rule compute_mahalanobis_gif:
    input: "{dirname}/mahalanobis.mat", "{dirname}/head_affine_registered.obj"
    output: "{dirname}/mahalanobis.gif"
    threads: 1
    run:
        MAT_PATH = r"C:\Users\h492884\PycharmProjects\mouseSkull\matlab"
        cmd= r""
        cmd += f'matlab -nosplash -noFigureWindows -sd {MAT_PATH} -batch "create_mah_gif('
        cmd+= f"'{input[0]}', '{input[1]}', '{output[0]}'); exit;" + '"'

        print(cmd)
        #shell(cmd)
        subprocess.call(cmd)
        #time.sleep(50)

rule compute_percentile_gif:
    input: "{dirname}/mahalanobis.mat", "{dirname}/head_affine_registered.obj"
    output: "{dirname}/percentile.gif"
    threads: 1
    run:
        par_name = Path(input[0]).parent.name
        m_or_f = par_name.split("_")[1]# copy males or females
        percentile_thr_file = MALE_FACE_PERC
        if m_or_f == "F":
            percentile_thr_file = FEMALE_FACE_PERC
        MAT_PATH = r"C:\Users\h492884\PycharmProjects\mouseSkull\matlab"
        cmd= r""
        cmd += f'matlab -nosplash -noFigureWindows -sd {MAT_PATH} -batch "create_perc_gif('
        cmd+= f"'{input[0]}', '{percentile_thr_file}' ,'{input[1]}', '{output[0]}'); exit;" + '"'

        print(cmd)
        #shell(cmd)
        subprocess.call(cmd)
        #time.sleep(50)



