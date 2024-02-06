import copy
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import List, Dict

import matplotlib
import vtk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import scipy.io as scio
from tqdm.auto import tqdm
import scipy.spatial as sp
import sys
print(__file__)
sys.path.append(  str(Path(workflow.basedir).parent.parent / 'scripts'))
from gpa import gpa
from vtk_helper import center_of_mesh
PROCESS_DIR = Path(config['process_dir'])
print(PROCESS_DIR)
#pathlib.Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir')
REF_IMAGE = Path(r'\\primus.img.cas.cz\data\83_BIOINFORMATICS\Igor\reference_gut')

ref_im = REF_IMAGE / 'oriented_image.nii.gz'
ref_mask = REF_IMAGE / 'bone_coregistration_mask.nii.gz'
coh_file  = r"\\primus.img.cas.cz\data\83_BIOINFORMATICS\Igor\reference_gut\ref.pkl"

# MALE_FACE_PERC = r"\\primus.img.cas.cz\data\83_BIOINFORMATICS\Igor\workdir\wt_male_f_percentiles.mat"
PERCENT = r"\\primus.img.cas.cz\data\83_BIOINFORMATICS\Igor\reference_gut\face_percentiles.mat"
# coh_file  = r"C:\Users\Public\wt_cohorts_pts.pkl"



def read_coh_file():
    coh_file_l = open(coh_file,'rb')
    COHORTS = pickle.load(coh_file_l)

    coh_file_l.close()
    return COHORTS
    print("FILE READ")

MICE_FOLDERS = [x for x in PROCESS_DIR.iterdir() if (x.is_dir() #and "Trab" in x.name
                and (x / "head_affine_registered.obj").exists())]
print("START BUILD")
from sklearn.covariance import EmpiricalCovariance,MinCovDet


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

def read_mesh_and_points( obj_filename ):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_filename)
    reader.Update()
    mesh = reader.GetOutput()
    points = mesh.GetPoints()

    res_mesh = []
    for i in range(points.GetNumberOfPoints()):
        pt = points.GetPoint(i)
        res_mesh.append(pt)

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

    return np.array(res_mesh), faces, mesh



def create_mah_gif(input_mat_file, input_obj_file, output_gif_file):
    # Load the energies
    cmp_str = 'jet'
    energies = scio.loadmat(input_mat_file)['energies']
    energies[np.where(energies > 100)] = 100
    energies = energies[0]
    print(energies[0])
    # Load the mesh
    pts,faces,mesh = read_mesh_and_points(input_obj_file)

    # Create a figure
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = fig.add_subplot(111, projection='3d')


    # Set the colormap (adjust as desired)
    norm = matplotlib.colors.Normalize(0, 100)
    m = cm.ScalarMappable(cmap=plt.cm.get_cmap(cmp_str),norm=norm)
    m.set_array([])
    print((norm(energies)))
    plt.colorbar(m)

    cmap = plt.get_cmap(cmp_str)
    norm = plt.Normalize(0, 100)
    cmap = plt.get_cmap(cmp_str)
    p3dc= ax.plot_trisurf(pts[:, 0],pts[:, 1],pts[:, 2],
        triangles=faces,linewidth=0.2  #,)
    )
    ax.set_box_aspect([np.ptp(i) for i in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    p3dc.set_fc(cmap(norm(energies)))
    #ax.set_autoscale_on(False)
    plt.savefig(output_gif_file.split(".")[0] + ".png")
    # Create the animation function
    def animate(frame):
        #ax.clear()
        #ax.set_box_aspect([1, 1, 1])
        ax.set_box_aspect([np.ptp(i) for i in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
        p3dc = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                        triangles=faces,linewidth=4)#,)

        p3dc.set_fc(cmap(norm(energies)))
        p3dc.set_lw(10)
        # ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
        #                 triangles=faces,linewidth=0.2, cmap=cmap, norm=norm,
        #                 facecolors=cmap(norm(energies))
        # )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame + 1}')

        # Rotate the view
        ax.view_init(elev=20,azim=frame * 36)  # Adjust the rotation speed (36 degrees per frame)

    # Number of frames in the GIF
    num_frames = 10

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=num_frames)

    # Save the animation as a GIF
    ani.save(output_gif_file, writer='pillow',fps=1 )

    plt.close(fig)
    print('GIF saved successfully.')

rule all:
    input:
        expand("{dirname}/mahalanobis.gif",dirname=MICE_FOLDERS)
        ,expand("{dirname}/percentile.gif", dirname=MICE_FOLDERS)

rule compute_mahalanobis_heads:
    input: "{dirname}/head_affine_registered.obj"
    output: "{dirname}/mahalanobis.mat"
    threads: 1
    run:
        COHORTS = read_coh_file()
        cohort = COHORTS
        par_name = Path(input[0]).parent.name

        mesh_pts,faces, mesh = read_mesh_and_points(input[0])


        com = center_of_mesh(mesh)
        cohort[par_name] = mesh_pts - np.array(com)

        #test_V = np.stack(cohort.values())
        _,_,shapes = gpa(cohort ,max_iterations=100, threshold=1e-11)

        art = shapes[par_name]
        shapes.pop(par_name)
        dists = single_to_many(mesh_pts=art,cohort=list(shapes.values()))

        # analyse distances


        ### compute colors and write to mat
        energies = []
        for x in faces:
            en = 0
            for p in x:
                en += dists[p]
            en = en / 3
            energies.append(en)
        scio.savemat(output[0],{"energies": energies})


# rule compute_mahalanobis_gif:
#     input: "{dirname}/mahalanobis.mat", "{dirname}/head_affine_registered.obj"
#     output: "{dirname}/mahalanobis.gif"
#     threads: 1
#     run:
#
#         create_mah_gif(input_mat_file=input[0],input_obj_file=input[1],output_gif_file=output[0])

rule compute_mahalanobis_fig:
    input: "{dirname}/mahalanobis.mat", "{dirname}/head_affine_registered.obj"
    output: "{dirname}/mahalanobis.fig"
    threads: 1
    run:
        MAT_PATH = r"C:\Users\h492884\PycharmProjects\mouseSkull\matlab"
        cmd= r""
        cmd += f'matlab -nosplash -noFigureWindows -sd {MAT_PATH} -batch "create_mah_fig('
        cmd+= f"'{input[0]}', '{input[1]}', '{output[0]}', 30); exit;" + '"'
        print(cmd)
        subprocess.call(cmd)

rule compute_mahalanobis_gif:
    input: "{dirname}/mahalanobis.fig"
    output: "{dirname}/mahalanobis.gif"
    threads: 1
    run:
        MAT_PATH = r"C:\Users\h492884\PycharmProjects\mouseSkull\matlab"
        cmd= r""
        cmd += f'matlab -nosplash -noFigureWindows -sd {MAT_PATH} -batch "fig_2_gif('
        cmd+= f"'{input[0]}', '{output[0]}'); exit;" + '"'

        print(cmd)
        subprocess.call(cmd)

rule compute_percentile_fig:
    input: "{dirname}/mahalanobis.mat", "{dirname}/head_affine_registered.obj"
    output: "{dirname}/percentile.fig"
    threads: 1
    run:
        par_name = Path(input[0]).parent.name
        percentile_thr_file = PERCENT

        MAT_PATH = r"C:\Users\h492884\PycharmProjects\mouseSkull\matlab"
        cmd= r""
        cmd += f'matlab -nosplash -noFigureWindows -sd {MAT_PATH} -batch "create_perc_fig('
        cmd+= f"'{input[0]}', '{percentile_thr_file}' ,'{input[1]}', '{output[0]}'); exit;" + '"'

        print(cmd)
        #shell(cmd)
        subprocess.call(cmd)


        #time.sleep(50)


rule compute_perc_gif:
    input: "{dirname}/percentile.fig"
    output: "{dirname}/percentile.gif"
    threads: 1
    run:
        MAT_PATH = r"C:\Users\h492884\PycharmProjects\mouseSkull\matlab"
        cmd= r""
        cmd += f'matlab -nosplash -noFigureWindows -sd {MAT_PATH} -batch "fig_2_gif('
        cmd+= f"'{input[0]}', '{output[0]}'); exit;" + '"'

        print(cmd)
        subprocess.call(cmd)
