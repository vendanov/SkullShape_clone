from typing import Union
import numpy as np
from mouse_skull_pipelines.scripts.mesh_algorithms import save_landmark_elastix
from mouse_skull_pipelines.scripts.nets import PointNetRegressor
import torch
from pathlib import Path
import vtk

net = PointNetRegressor(num_points=1000, num_outputs=9, input_dim=3)
net.load_state_dict(torch.load("/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/pointnet_model_cp_detection.pth"))


ref_path = "/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/reference_image"
ref_mask = Path(ref_path) / "oriented_mask.nii.gz"


pts = net.compute_from_image(ref_mask)
print("output points", pts)
save_landmark_elastix(Path(ref_path) / 'ref_landmarks.pts', pts)