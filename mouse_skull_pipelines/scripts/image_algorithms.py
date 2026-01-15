import shutil
import subprocess
from dataclasses import dataclass
from typing import Union, Tuple

import PIL.Image as pim
import vtk
import numpy as np
from pathlib import Path
import SimpleITK as sitk




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

def create_oriented_bounding_box(mesh: vtk.vtkPolyData):
    """
    Create an Oriented Bounding Box (OBB) from the given mesh.
    Args:
        mesh: vtkPolyData object.

    Returns:
        - Oriented bounding box (OBB) transform.
        - Corner point of the OBB.
        - Half-lengths along each axis.
        - Directions (axes of the OBB).
    """
    obb_tree = vtk.vtkOBBTree(data_set=mesh, max_level=1)
    obb_tree.BuildLocator()

    # Allocate the variables needed for OBB
    corner = [0.0, 0.0, 0.0]
    max_point = [0.0, 0.0, 0.0]
    mid_point = [0.0, 0.0, 0.0]
    min_point = [0.0, 0.0, 0.0]
    size = [0.0, 0.0, 0.0]

    # Compute the OBB
    obb_tree.ComputeOBB(mesh,corner, max_point, mid_point, min_point,size)

    # Calculate OBB directions (axes)
    axes = [
        np.array(max_point) - np.array(corner),
        np.array(mid_point) - np.array(corner),
        np.array(min_point) - np.array(corner)
    ]
    axes = [axis / np.linalg.norm(axis) for axis in axes]  # Normalise the axes

    # Create the OBB transform
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(corner)

    # Set rotation using the axes
    matrix = transform.GetMatrix()
    for i, axis in enumerate(axes):
        for j in range(3):
            matrix.SetElement(j, i, axis[j])
    transform.SetMatrix(matrix)

    return transform, size

def crop_image_with_obb(input_file, output_file, obb_transform, obb_lengths):
    """
    Crop an image using an Oriented Bounding Box (OBB).
    Args:
        input_file: Path to the input NIfTI image file.
        output_file: Path to the output NIfTI image file.
        obb_transform: vtkTransform defining the OBB.
        obb_lengths: Half-lengths along each OBB axis.
    """
    # Read the input image
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(input_file))
    reader.Update()
    image = reader.GetOutput()

    # Create an implicit box function representing the OBB
    box = vtk.vtkBox()
    box.SetBounds(
        -obb_lengths[0], obb_lengths[0],
        -obb_lengths[1], obb_lengths[1],
        -obb_lengths[2], obb_lengths[2]
    )

    # Apply the OBB transform to the box
    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetTransform(obb_transform)
    transform_filter.SetInputConnection(reader.GetOutputPort())
    transform_filter.Update()

    # Use vtkImageReslice for extracting the ROI defined by the OBB
    reslice = vtk.vtkImageReslice()
    reslice.SetInputConnection(reader.GetOutputPort())
    reslice.SetResliceTransform(obb_transform.GetInverse())
    reslice.SetOutputExtent(-obb_lengths[0], obb_lengths[0], -obb_lengths[1], obb_lengths[1], -obb_lengths[2], obb_lengths[2])
    reslice.Update()

    # Write the cropped image
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(str(output_file))
    writer.SetInputConnection(reslice.GetOutputPort())
    writer.Write()


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


def read_sitk_image( file_name: Union[Path, str]) -> sitk.Image:
    """
    Read an image file using SimpleITK.
    Args:
        file_name: The path to the image file.

    Returns:
        The SimpleITK image object.
    """
    return sitk.ReadImage(str(file_name))


def generate_projection_image(input_image: sitk.Image, axis=1):
    """
    Generate a maximum intensity projection image along the specified axis.
    Args:
        input_image: The input 3D image.
        axis: The axis along which to project (0, 1, or 2).

    Returns:
        The projection image array.
    """
    # Create the projection filter
    projection_filter = sitk.MaximumProjectionImageFilter()
    projection_filter.SetProjectionDimension(axis)

    # Execute the projection filter
    projection_image = projection_filter.Execute(input_image)

    projection_image_2d = sitk.GetArrayFromImage(projection_image)

    if axis == 0:
        projection_image_2d = projection_image_2d[:, :, 0]
    elif axis == 1:
        projection_image_2d = projection_image_2d[:, 0, :]
    elif axis == 2:
        projection_image_2d = projection_image_2d[0, :, :]
    else:
        raise ValueError("Invalid axis value. Must be 0, 1, or 2.")

    return projection_image_2d

def convert_2d_to_PIL(image: np.ndarray) -> pim:

    flo_proj_im = image
    flo_proj_im = (flo_proj_im - np.min(flo_proj_im)) / (np.max(flo_proj_im) - np.min(flo_proj_im))  # Normalize to 0-1
    flo_proj_im = (flo_proj_im * 255).astype(np.uint8)  # Convert to 8-bit integer
    flo_proj_im = pim.fromarray(flo_proj_im)
    return flo_proj_im


def compute_2d_transform_no_opencv(source_points, target_points):
    """
    Compute the 2D affine transformation matrix (rotation + translation)
    that aligns `source_points` to `target_points` without using OpenCV.
    """
    # Calculate centroids of the source and target lines
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Center the points around the centroid
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Calculate the rotation matrix using Singular Value Decomposition (SVD)
    H = np.dot(source_centered.T, target_centered)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Ensure proper rotation (determinant must be +1)
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the translation vector
    T = target_centroid - np.dot(R, source_centroid)

    # Combine into a 2D affine transformation matrix
    transform_matrix = np.eye(3)
    transform_matrix[:2, :2] = R
    transform_matrix[:2, 2] = T

    return transform_matrix


def convert_2d_to_3d_transform_no_opencv(transform_2d, projection_axis):
    """
    Convert a 2D affine transformation into a 3D affine transformation matrix.
    """
    transform_3d = np.eye(4)  # Initialize a 4x4 identity matrix
    axes = [0, 1, 2]
    axes.remove(projection_axis)  # Identify the 2D plane (e.g., XY if Z is the projection axis)

    # Embed the 2D transformation matrix into the 3D matrix
    transform_3d[np.ix_(axes, axes)] = transform_2d[:2, :2]  # Rotation
    transform_3d[np.ix_(axes, [3])] = transform_2d[:2, 2].reshape(-1, 1)  # Translation

    return transform_3d


@dataclass
class LinearRegistrationConfig:
    is_simple: bool = False
    ref_landmarks: str = None
    side_landmarks_model: str = None
    side_reference_xml: str = None
    control_points_model: str = None



def register_linear_full(floating_image: Union[str, Path],
                         reference_image: Union[str, Path],
                         float_mask: Union[str, Path],
                         ref_mask: Union[str, Path],
                         outputs: Tuple[str, str],
                         config: LinearRegistrationConfig,
                         inaff=None):
    """

    :param floating_image:
    :param reference_image:
    :param float_mask:
    :param ref_mask:
    :param outputs:
    :param inaff:
    :return:
    """
    from .nets import coalign_roi_based

    metrics_outputs = {}

    process_folder = Path(floating_image).parent
    res_mask_path = process_folder / "coregistered_mask.nii.gz"
    if config.is_simple:

        metric = perform_linear_coregistration(
            floating_image=floating_image, reference_image= reference_image,
            ref_mask_=ref_mask,
            float_mask=float_mask,outputs= outputs, config= config, inaff=inaff, use_landmarks=False)
        return metric

    if coalign_roi_based(reference_image,floating_image,
        ref_rois=config['side_ref_xml'],model_path=config['side_model'],mask_path=float_mask,
        res_path=outputs[0],
        res_mask_path=res_mask_path,out_im_name='axis_0.png',axis=0) is not None:

        metric_0 = perform_linear_coregistration(
            outputs[0],reference_image,config.ref_landmarks,ref_mask,res_mask_path,outputs,inaff,use_landmarks=False)
    else:
        metric_0 = perform_linear_coregistration(
            floating_image,reference_image,config.ref_landmarks,ref_mask,float_mask,outputs,inaff,use_landmarks=False)
    #metrics_outputs.append(metric_0)
    print("Metric 0",metric_0)

def compute_affine_transformation(a_points, b_points):
    """
    Compute the affine transformation matrix that maps a_points to b_points.

    Parameters:
    - a_points: List or array of three source points (3x3 matrix).
    - b_points: List or array of three destination points (3x3 matrix).

    Returns:
    - A 4x4 homogeneous transformation matrix.
    """
    a_points = np.array(a_points)
    b_points = np.array(b_points)

    # Compute centroids
    centroid_a = np.mean(a_points,axis=0)
    centroid_b = np.mean(b_points,axis=0)

    # Subtract centroids to center the points
    a_centered = a_points - centroid_a
    b_centered = b_points - centroid_b

    # Compute covariance matrix
    H = a_centered.T @ b_centered

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected, adjusting for proper rotation.")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Compute translation vector
    t = centroid_b - R @ centroid_a

    # Construct homogeneous transformation matrix
    homogeneous_matrix = np.identity(4)
    homogeneous_matrix[:3, :3] = R
    homogeneous_matrix[:3, 3] = t

    print("Hom matrix",homogeneous_matrix)

    return homogeneous_matrix
import nibabel as nib

def update_image_affine(input_ct, output_head, affine_matrix, copy=False, copy_im=None):
    if copy:
        image_f = nib.load(copy_im)
        image_affine = image_f.affine
        old_im = nib.load(input_ct)

        new_image = nib.Nifti1Image(old_im.get_fdata(),image_affine)
        print("Image affine",input_ct,image_affine)

        nib.save(new_image,output_head)
        return

    image_f = nib.load(input_ct)
    image_affine = image_f.affine
    print("Image affine",input_ct,image_affine)

    new_affine = affine_matrix @ image_affine

    # Update the image's affine matrix
    new_image = nib.Nifti1Image(image_f.get_fdata(),new_affine,image_f.header)
    nib.save(new_image,output_head)



def measure_coregistration_metric(image_path1, image_path2, metric="correlation"):
    """
    Measure coregistration metric between two images.

    Parameters:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.
        metric (str): Metric to evaluate. Options: "mutual_information", "correlation".

    Returns:
        float: Metric value representing similarity between the images.
    """
    # Load the images
    image1 = sitk.ReadImage(image_path1,sitk.sitkFloat32)
    image2 = sitk.ReadImage(image_path2,sitk.sitkFloat32)

    # Convert images to numpy arrays
    image1_array = sitk.GetArrayFromImage(image1)
    image2_array = sitk.GetArrayFromImage(image2)

    # Compute the metric
    if metric == "mutual_information":
        joint_histogram, _, _ = np.histogram2d(image1_array.ravel(),image2_array.ravel(),bins=32)
        joint_prob = joint_histogram / np.sum(joint_histogram)
        marginal_prob1 = np.sum(joint_prob,axis=0)
        marginal_prob2 = np.sum(joint_prob,axis=1)
        mutual_information = np.sum(
            joint_prob[joint_prob > 0] * np.log(joint_prob[joint_prob > 0] /
                                                (marginal_prob1[np.newaxis, :] *
                                                 marginal_prob2[:, np.newaxis])[joint_prob > 0]))
        return mutual_information

    elif metric == "correlation":
        correlation = np.corrcoef(image1_array.ravel(),image2_array.ravel())[0, 1]
        return correlation

    else:
        raise ValueError(f"Unsupported metric: {metric}")




def perform_linear_coregistration(floating_image, reference_image,
                                  ref_mask_, float_mask, outputs, config,
                                  inaff=None, use_landmarks=True):
    proces_folder = Path(floating_image).parent
    out_file = Path(outputs[0])
    # landmark generation
    landmark_file = proces_folder / "landmarks.pts"
    from .mesh_algorithms import load_landmark_elastix,save_landmark_elastix
    from .nets import PointNetRegressor
    if use_landmarks:
        pts = PointNetRegressor.landmark_point_generation(ref_mask_,   config.control_points_model)
        save_landmark_elastix(landmark_file, pts)
        fix_pts_np = load_landmark_elastix(config.ref_landmarks)
        mov_pts_np = load_landmark_elastix(landmark_file)
        # write alignment script
        aff_mat = compute_affine_transformation(mov_pts_np, fix_pts_np)
        update_image_affine(floating_image, output_head=out_file, affine_matrix=aff_mat)
        update_image_affine(float_mask, output_head=proces_folder / "coregistered_mask.nii.gz",
                            affine_matrix=aff_mat, copy=False, copy_im=floating_image)
    else:
        aff_mat = np.eye(4)
        def copy_image(inp, out):
            shutil.copy2(str(inp), str(out))
        copy_image(floating_image, out_file)
        copy_image(float_mask, str(proces_folder / "coregistered_mask.nii.gz"))
        # update_image_affine(floating_image, output_head=out_file, affine_matrix=aff_mat)
        # update_image_affine(float_mask, output_head=proces_folder / "coregistered_mask.nii.gz",
        #                     affine_matrix=aff_mat, copy=True, copy_im=floating_image)

    np.savetxt(proces_folder / "rigid_head.txt", aff_mat)

    if Path(float_mask).name == "coregistered_mask.nii.gz":
        print("Mask is already coregistered")
    # sys.exit(-2)
    # linear coregistration
    register_flip(floating_image, reference_image, ref_mask=ref_mask_, float_mask=float_mask, outputs=outputs,
                  inaff=inaff)
    metric = measure_coregistration_metric(outputs[0], reference_image, metric="correlation")
    return metric



def otsu_treshold(input_file: Path, out_file: Path, origin_copy=None):
    # Load image
    img = sitk.ReadImage(input_file)

    # Apply Otsu thresholding
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    otsu_img = otsu_filter.Execute(img)
    if origin_copy is not None:
        orcp = sitk.ReadImage(origin_copy)
        otsu_img.SetOrigin(orcp.GetOrigin())
        otsu_img.SetSpacing(orcp.GetSpacing())
        otsu_img.SetDirection(orcp.GetDirection())
    else:
        otsu_img.SetOrigin(img.GetOrigin())
        otsu_img.SetSpacing(img.GetSpacing())
        otsu_img.SetDirection(img.GetDirection())

    # Save thresholded image
    sitk.WriteImage(otsu_img,out_file)


def register_flip(floating_image: Union[str, Path],
                  reference_image: Union[str, Path], float_mask: Union[str, Path],
                  ref_mask: Union[str, Path],
                  outputs: Tuple[str, str],
                  inaff=None):
    """
    :param floating_image:
    :param reference_image:
    :param float_mask:
    :param ref_mask:
    :param outputs:
    :param inaff:
    :return:
    """
    if inaff is None:
        # cmd = ['reg_aladin', '-ref',
        #        str(reference_image), '-rmask', ref_mask,
        #        '-flo', floating_image, '-fmask', float_mask,
        #        '-speeeed', '-cog', '-res', str(outputs[0]), '-aff', str(outputs[1])]
        cmd = "reg_aladin -ref '{refim}' -rmask '{rmask}' -flo '{flim}' -fmask '{fmask}'".format(refim=reference_image,
            rmask=ref_mask,flim=floating_image,fmask=float_mask)
        cmd += " -speeeeed -cog -res '" + str(outputs[0])
        cmd += "' -aff '" + outputs[1] + "'"
        print(cmd)
        subprocess.run(cmd, shell=True)
        #shell(cmd)

def non_rigid_registration(flo_im: str, ref_im: str, out_transform: str, out_image: str, param_file: str):
    # Reading
    def shell(cmd: str):
        subprocess.run(cmd, shell=True, check=True)

    fold = Path(out_image).parent
    cmd = f"elastix -f '{ref_im}' -m '{flo_im}' -p '{param_file}' -out '{str(fold)}'"
    shell(cmd)
    shell(f"mv -f '{str(fold / 'result.0.nii.gz')}' '{out_image}'")
    shell(f"mv -f '{str(fold / 'TransformParameters.0.txt')}' '{out_transform}'")
