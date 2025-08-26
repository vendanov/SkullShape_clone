#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pickle
import sys
sys.path.append('../mouse_skull_pipelines/scripts')
import gpa
import vtk_helper
import numpy as np
from typing import Dict,List
from tqdm.auto import tqdm
from sklearn.covariance import EmpiricalCovariance, EllipticEnvelope, MinCovDet
import pandas as pd

from scipy.io import savemat
import vtk


# In[5]:


mouse_wild_types = '/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir'

mesh_file =   "/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/reference_image/oriented_mesh.obj"

#outputs
perc_out =   "/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/wt_mah_percentiles.pkl"
perc_out_male =   "/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/wt_male_percentiles.mat"
perc_out_male_face =   "/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/wt_male_f_percentiles.mat" # important output for now!
perc_out_female = "/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/wt_female_percentiles.mat" # 
perc_out_female_face =   "/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/wt_female_f_percentiles.mat" # important output for now! 


# In[6]:


o_f =   "/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/JUW_3W_pts.pkl"
of = open(o_f,'rb')
male_female = pickle.load(of)
of.close()


# In[7]:


# mahalanobises

results_groups = {k: gpa.gpa(x, 100, 10e-11)[2] for k,x in male_female.items()}

# _,_,res_males = gpa(male_female['M'], 100, 10e-11)
# _,_,res_females = gpa(male_female['F'], 100, 10e-11)
    
    
    
    


# In[ ]:


def mahalanobis_distibutions(cohort: Dict[str,np.ndarray]) -> List[np.ndarray]:
    # build distributions
    num_of_points = list(cohort.values())[0].shape[0]
    dists = []
    for i in tqdm(range(num_of_points)):  # i in range of num_of_points
        points = []
        #print (i, working_dict)

        for mesh_v in cohort.values():
            points.append(mesh_v[i])

        points = np.vstack(points)
        #print(points)
        emp_cov = EmpiricalCovariance().fit(points)
        
        mah_dists = []
        for point in range(points.shape[0]):
            dst = emp_cov.mahalanobis([points[point]])
            mah_dists.append(dst)
        threshold = np.percentile(mah_dists, 99)
        
        #print(mah_dists)
        #mah_dists = np.array(mah_dists)[np.where(np.array(mah_dists) < 10)[0]]
        dists.append(np.array(mah_dists))
        #break
        
    return dists




groups_res = { k : mahalanobis_distibutions(v) for k,v in results_groups.items()}
# res_mahalanobis_males =  mahalanobis_distibutions(res_males)
# res_mahalanobis_females = mahalanobis_distibutions(res_females)


# In[ ]:

# In[ ]:


def get_percentiles_faces(percentiles : Dict[int,List[float]], obj_filename) -> Dict[int,List[float]]:
    
    mesh = vtk_helper.read_vtk_obj(obj_filename)
    
    faces = vtk_helper.get_mesh_faces(mesh)
    
    face_percentiles = dict(zip(percentiles.keys(),[[] for x in percentiles.keys()])) 
        

    for face in tqdm( faces):
        res_dict_tmp =  dict(zip(percentiles.keys(),[0.0 for x in percentiles.keys()])) 
        
        for vert_id in face:
            
            # compute sum for all vertices in faces
            for key in percentiles.keys():
                res_dict_tmp[key] = res_dict_tmp[key] + percentiles[key][vert_id]
            
        # compute average and sum
        for key in percentiles.keys():
            face_percentiles[key].append (  res_dict_tmp[key] /3 )
    return face_percentiles


# In[ ]:


def get_percentiles(mah_distancess : List[np.ndarray] ) -> Dict[int,List[float]]:
    #computes percentiles for points
    ###99 percentiles 
    percentiles = {75:[], 95:[], 99:[]}

    for mahalanobis in tqdm(mah_distancess):
        for perc in percentiles.keys():
            percentiles[perc].append(np.percentile(mahalanobis, perc,interpolation='higher'))
    return percentiles




perc_group = { k : get_percentiles(v) for k,v in groups_res.items()}
# perc_fem = get_percentiles(res_mahalanobis_females)
# perc_m = get_percentiles(res_mahalanobis_males)



# In[ ]:


###############write out 
out_file = open(perc_out,"wb")
pickle.dump( perc_group,out_file)
out_file.close()


# In[ ]:

#
# savemat(perc_out_female,{ "perc"+str(k): v for k,v in perc_fem.items()})
# savemat(perc_out_male,{ "perc" + str(k): v for k,v in perc_m.items()})


# In[ ]:


###compute average percentiles for faces!!

perc_f_group = { k : get_percentiles_faces(v,obj_filename=mesh_file) for k,v in perc_group.items()}


for k,v in perc_f_group.items():
    savemat("/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/juw3w"+k+"f_percentiles.mat",{ "perc" + str(k): v for k,v in v.items()})

# savemat(perc_out_male_face,{ "perc" + str(k): v for k,v in perc_out_male_f.items()})
# savemat(perc_out_female_face,{ "perc"+str(k): v for k,v in perc_out_female_f.items()})





