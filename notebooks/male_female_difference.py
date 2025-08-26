#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import vtk
from pathlib import Path
from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy import io as scio
from scipy import stats as sci_stats
import pickle
import scipy.spatial as sp
from typing import Union,Dict,List
import copy
import sys
sys.path.append('../mouse_skull_pipelines/scripts')
import gpa


# In[4]:


path_to_processed_cohort = r"/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/JUVENILE_STUDY/WT_w03/skull_shape_workdir"
result_energies = r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/juv_3W.mat"

# In[5]:


def compute_center_of_mesh(mesh):
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(mesh)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()
    center = centerOfMassFilter.GetCenter()
    return center


reader = vtk.vtkOBJReader()
num_of_points = 0
meshes_dict = {}
for path in tqdm(list(Path(path_to_processed_cohort).iterdir())):
    mesh_file = path / "head_affine_registered.obj"
    print(mesh_file)
    if not mesh_file.exists():
        continue
    reader.SetFileName(mesh_file)
    reader.Update()
    mesh = reader.GetOutput()
    points = mesh.GetPoints()
    center_of_mesh = np.array( compute_center_of_mesh(mesh) )
    num_of_points = points.GetNumberOfPoints()
    res_mesh = []
    for i in range(points.GetNumberOfPoints()):
        pt = points.GetPoint(i)
        res_mesh.append(list( np.array( pt) - center_of_mesh ))
    meshes_dict[str( mesh_file.parent.name ) ] = res_mesh

# In[ ]:


def write_obj_2_file(obj,filename):
    f = open(filename,"wb")
    pickle.dump(obj,f)
    f.close()

def compute_descriptive_statistics(points):
    mean = np.mean(points, axis=0)
    std_dev = np.std(points, axis=0)
    # Add any other statistics you're interested in
    return mean, std_dev

def compute_outliers(meshes_dict: Union[Dict[str,np.ndarray],List[np.ndarray]]):
    
    stats = []
    index_sum=None
    if isinstance(meshes_dict,list):
        print(len(meshes_dict))
        num_of_points = len( meshes_dict[0] )
        working_dict = list(zip(range(num_of_points),meshes_dict))
        
    else:
        print(len(meshes_dict))
        num_of_points = len( list(meshes_dict.values())[0] )
        working_dict = list(meshes_dict.items())
            
    for i in tqdm(range(num_of_points)): # i in range of num_of_points
        index = []
        points = []
        #print (i, working_dict)
        for key, mesh_v in working_dict:
            index.append(key)
            points.append(mesh_v[i])
        
        df = pd.DataFrame( data = points, index=index, columns=['x','y','z'])
        
        indexes = (np.abs(sci_stats.zscore(df)) < 2).all(axis=1)
        
        if index_sum is None:
            index_sum = (~indexes).astype(int)
        else:
            index_sum += (~indexes).astype(int)
        df = df[indexes]
        stats.append(df.describe())
    
    return stats, index_sum

    

stats, index_sum = compute_outliers(meshes_dict)



# In[6]:


meshes_dict_copy = copy.copy(meshes_dict)


#copy dict 
# for k,v in sorted(index_sum.items(), key=lambda x:x[1]):
#     if v > 1000:
#         meshes_dict_copy.pop(k)


# In[ ]:





# In[7]:


#procrustes analysis

ref_mesh, aligned_meshes, shapes= gpa.gpa(meshes_dict_copy,max_iterations=100, threshold=1e-11)


# In[11]:


#divide into unique groups by
unique_groups = set([ x.split("_")[1] for x in shapes.keys() ])

grouped_shapes = {}
for k, arr in shapes.items():
    group_label = k.split("_")[1]
    if group_label not in grouped_shapes:
        grouped_shapes[group_label] = {}
    grouped_shapes[group_label][k] = arr



# In[ ]:


# SAVE MALE_FEMALE COHORT

# In[9]:


grouped_shapes = {}
for k, arr in meshes_dict_copy.items():
    group_label = k.split("_")[1]
    if group_label not in grouped_shapes:
        grouped_shapes[group_label] = {}
    grouped_shapes[group_label][k] = arr


o_f =   "/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/JUW_3W_pts.pkl"
of = open(o_f,'wb')
pickle.dump(grouped_shapes,of)
of.close()

# In[ ]:




