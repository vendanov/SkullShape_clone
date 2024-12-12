from typing import Union

import vtk
import numpy as np
from pathlib import Path
import torch
from pytorch3d.ops import sample_farthest_points


# Algorithms for image processing using VTK and NumPy
# No neural network code here, just image processing and pre-processing


def sample_points_from_mesh(mesh: vtk.vtkPolyData, point_count=1000):
    """
    Sample points from a vtk mesh object using the farthest point sampling algorithm.
    preserve the original mesh structure
    Args:
        mesh_file:
        point_count:

    Returns:

    """
    origin_mesh = mesh
    origin_pts = origin_mesh.GetPoints()
    origin_pts = np.array([origin_pts.GetPoint(i) for i in range(origin_pts.GetNumberOfPoints())])
    # compute the from origin_pts
    # convert np array to torch tensor
    origin_pts = torch.tensor(origin_pts, dtype=torch.float32)
    res_pts, _ = sample_farthest_points(origin_pts.unsqueeze(0), None, point_count)
    res_pts = res_pts.squeeze(0).numpy()
    return res_pts


def load_landmark_elastix(filename, coordinate_system='LPS'):
    """
    Load control points from an Elastix .pts file and convert them to the specified coordinate system.

    Parameters:
    - filename (str): Path to the .pts file.
    - coordinate_system (str): 'LPS' or 'RAS'. Default is 'LPS'.

    Returns:
    - points (numpy.ndarray): Array of shape [K, 3] containing control points.
    """
    with open(filename, 'r') as f:
        # Read the header lines and number of points
        header = f.readline().strip()  # "index"
        num_points = int(f.readline().strip())  # Number of points

        # Initialize an array to store the points
        points = np.zeros((num_points, 3))

        # Read each point
        for idx in range(num_points):
            line = f.readline().strip().split()
            points[idx] = np.array([float(coord) for coord in line])

    # Convert from RAS to LPS if necessary
    if coordinate_system.upper() == 'RAS':
        points = convert_LPS_to_RAS(points)
    elif coordinate_system.upper() != 'LPS':
        raise ValueError("Coordinate system must be either 'LPS' or 'RAS'")

    return points

def save_landmark_elastix(filename, points, coordinate_system='LPS'):
    """
    Save control points to an Elastix .pts file in the specified coordinate system.

    Parameters:
    - filename (str): Path to the output .pts file.
    - points (numpy.ndarray): Array of shape [K, 3] containing control points.
    - coordinate_system (str): 'LPS' or 'RAS'. Default is 'LPS'.
    """
    if coordinate_system.upper() == 'RAS':
        # Convert from LPS to RAS if necessary
        points = convert_LPS_to_RAS(points)
    elif coordinate_system.upper() != 'LPS':
        raise ValueError("Coordinate system must be either 'LPS' or 'RAS'")

    K = points.shape[0]
    with open(filename, 'w') as f:
        f.write("index\n")
        f.write(f"{K}\n")
        for idx, point in enumerate(points):
            # Format: index x y z
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

def load_landmark_elastix(filename, coordinate_system='LPS'):
    """
    Load control points from an Elastix .pts file in the specified coordinate system.
    Args:
        filename:
        coordinate_system:

    Returns:

    """
    with open(filename, 'r') as f:
        # Read the header lines and number of points
        header = f.readline().strip()  # "index"
        num_points = int(f.readline().strip())  # Number of points

        # Initialize an array to store the points
        points = np.zeros((num_points, 3))

        # Read each point
        for idx in range(num_points):
            line = f.readline().strip().split()
            points[idx] = np.array([float(coord) for coord in line])

    # Convert from RAS to LPS if necessary
    if coordinate_system.upper() == 'RAS':
        points = convert_LPS_to_RAS(points)
    elif coordinate_system.upper() != 'LPS':
        raise ValueError("Coordinate system must be either 'LPS' or 'RAS'")

    return points




def convert_LPS_to_RAS(points):
    """
    Convert points from LPS to RAS coordinate system.

    Parameters:
    - points (numpy.ndarray): Array of shape [K, 3] in LPS.

    Returns:
    - numpy.ndarray: Array of shape [K, 3] in RAS.
    """
    ras_points = points.copy()
    ras_points[:, 0] = -ras_points[:, 0]  # Left <-> Right
    ras_points[:, 1] = -ras_points[:, 1]  # Posterior <-> Anterior
    return ras_points


def normalize_point_cloud(pc,centroid = None, scale=None):
    # pc: [N, 3]
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    if scale is None:
        scale = np.max(np.linalg.norm(pc, axis=1))

    pc1 = pc - centroid
    print("Centered point cloud: ", pc1)
    pc1 = pc1 / scale
    print("Normalized point cloud: ", pc1)
    return pc1, centroid, scale
