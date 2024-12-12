from typing import Union

import vtk
import numpy as np
from pathlib import Path


# Algorithms for image processing using VTK and NumPy
# No neural network code here, just image processing and pre-processing


def world_to_voxel(point, origin, spacing):
    """
    Convert a point in world coordinates to voxel coordinates.

    Parameters:
    - point: A 3D point in world coordinates [x, y, z].
    - origin: The origin of the image in world coordinates [ox, oy, oz].
    - spacing: The voxel spacing of the image in world coordinates [sx, sy, sz].

    Returns:
    - voxel_point: The 3D point in voxel coordinates.
    """
    voxel_point = [(point[i] - origin[i]) / spacing[i] for i in range(3)]
    return np.array(voxel_point)


def apply_transform_to_point(point, image_data,inverse=False):
    """
    Apply a vtkTransform to a 3D point.

    Parameters:
    - point: The 3D point in world coordinates (or any coordinate system) [x, y, z].
    - transform: The vtkTransform object containing the transformation.

    Returns:
    - transformed_point: The point after applying the transform.
    """

    transform = image_data[2]
    # Get the 4x4 matrix from the vtkTransform
    matrix = transform.GetMatrix()

    # Convert the vtkMatrix4x4 to a numpy array for easier manipulation
    transform_matrix = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
    if inverse:
        transform_matrix = np.linalg.inv(transform_matrix)
    print("transform_matrix: ", transform_matrix)
    # Convert the point to homogeneous coordinates [x, y, z, 1]
    if len(point.shape) > 1:
        transformed_points = np.zeros_like(point)
        for i in range(point.shape[0]):
            homogeneous_point = np.array([point[i,0], point[i,1], point[i,2], 1.0])
            # Apply the transformation matrix to the point
            transformed_point_homogeneous = transform_matrix.dot(homogeneous_point)

            # Convert back to 3D coordinates by removing the homogeneous component
            transformed_point = transformed_point_homogeneous[:3]
            transformed_points[i] = (transformed_point - np.array(image_data[0]))/np.array(image_data[1])

        return transformed_points



    homogeneous_point = np.array([point[0], point[1], point[2], 1.0])

    # Apply the transformation matrix to the point
    transformed_point_homogeneous = transform_matrix.dot(homogeneous_point)

    # Convert back to 3D coordinates by removing the homogeneous component
    transformed_point = transformed_point_homogeneous[:3]

    return transformed_point

def marching_cubes(input_file: Union[Path, str]) -> vtk.vtkPolyData:
    """
    Create a cleaned mesh from a thresholded image using the marching cubes algorithm.
    Args:
        input_file: Path to the thresholded image file
    Returns:

    """
    # Load the thresholded image vtk
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(input_file))
    reader.Update()

    # Extract the image's origin and spacing
    origin = reader.GetOutput().GetOrigin()
    spacing = reader.GetOutput().GetSpacing()

    # Create a marching cubes filter
    mc = vtk.vtkMarchingCubes()
    mc.SetInputConnection(reader.GetOutputPort())
    mc.SetValue(0, 1)
    mc.Update()

    origin = np.array(origin)
    spacing = np.array(spacing)


    transform = vtk.vtkTransform()
    if reader.GetQFormMatrix():
        transform.SetMatrix(reader.GetQFormMatrix())
    elif reader.GetSFormMatrix():
        transform.SetMatrix(reader.GetSFormMatrix())



    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(mc.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    # Get the aligned mesh
    aligned_mesh = transform_filter.GetOutput()

    # Remove small, unconnected meshes
    connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
    connectivity_filter.SetInputData(aligned_mesh)
    connectivity_filter.SetExtractionModeToAllRegions()
    connectivity_filter.ColorRegionsOn()
    connectivity_filter.Update()

    # Extract regions and remove small ones
    connectivity_filter.SetExtractionModeToSpecifiedRegions()
    connectivity_filter.InitializeSpecifiedRegionList()

    region_sizes = connectivity_filter.GetRegionSizes()

    for i in range(connectivity_filter.GetNumberOfExtractedRegions()):
        if region_sizes.GetValue(i) >= 100000:
            connectivity_filter.AddSpecifiedRegion(i)

    connectivity_filter.Update()

    # Create a cleaned mesh
    cleaned_mesh = vtk.vtkPolyData()
    cleaned_mesh.ShallowCopy(connectivity_filter.GetOutput())

    # Now remove points not included in the cleaned mesh
    cleaned_mesh.BuildCells()
    cleaned_mesh.BuildLinks()
    points = cleaned_mesh.GetPoints()
    num_points = points.GetNumberOfPoints()

    point_mask = np.zeros(num_points, dtype=bool)

    for i in range(cleaned_mesh.GetNumberOfCells()):
        cell = cleaned_mesh.GetCell(i)
        ids = cell.GetPointIds()
        for j in range(ids.GetNumberOfIds()):
            point_mask[ids.GetId(j)] = True

    new_points = vtk.vtkPoints()
    new_polydata = vtk.vtkPolyData()
    new_polydata.Allocate(cleaned_mesh.GetNumberOfCells())

    id_map = {}
    new_point_id = 0

    for i in range(num_points):
        if point_mask[i]:
            id_map[i] = new_point_id
            # convert the points to the LPS coordinate system
            pt  = points.GetPoint(i)
            #pt_lps = [-pt[0], -pt[1], pt[2]]# true lps
            pt_lps = pt
            new_points.InsertNextPoint(pt_lps)
            new_point_id += 1

    new_polydata.SetPoints(new_points)

    for i in range(cleaned_mesh.GetNumberOfCells()):
        cell = cleaned_mesh.GetCell(i)
        ids = cell.GetPointIds()
        new_cell_ids = vtk.vtkIdList()

        for j in range(ids.GetNumberOfIds()):
            new_cell_ids.InsertNextId(id_map[ids.GetId(j)])
        new_polydata.InsertNextCell(cleaned_mesh.GetCellType(i), new_cell_ids)

    # Update the polydata to remove unconnected points
    new_polydata.Modified()

    return new_polydata, (origin, spacing, transform)
