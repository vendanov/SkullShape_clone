import os
import os
import pickle
import shlex
import subprocess

import ants
import nibabel as nib
import sys
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Dict, Tuple, List
import torch
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
import vtk
from PIL import Image
from matplotlib import animation, cm
from sklearn.covariance import EmpiricalCovariance
from tqdm.auto import tqdm

from mouse_skull_pipelines.scripts.gpa import gpa, procreustes_alignment
from mouse_skull_pipelines.scripts.mesh_algorithms import save_landmark_elastix, load_landmark_elastix
from mouse_skull_pipelines.scripts.nets import PointNetRegressor, SuturesNetWrapper
from mouse_skull_pipelines.scripts.vtk_helper import *

PROCESS_DIR = Path(config['cohort_dir'])

REF_IMAGE = Path(config['ref_image_dir'])
PARAMETER_FILE_NONLIN = Path(config['parameter_file_nonlin'])
PARAMETER_FILE_LIN = Path(config['parameter_file_lin'])
MESH_TRANSFORM_SCRIPT = config['mesh_transform_script']
SLICER_CORE = config['slicer_core']

# Reference image paths
ref_im = REF_IMAGE / config['ref_im']
ref_mask = REF_IMAGE / config['ref_mask']
ref_landmarks = REF_IMAGE / config['ref_landmarks']
# Additional file paths
MALE_FACE_PERC = Path(config['male_face_perc'])
FEMALE_FACE_PERC = Path(config['female_face_perc'])
coh_file = Path(config['coh_file'])
control_points_model = Path(config['control_points_model'])

#####script for head conversion
@dataclass
class MouseDescriptor:

    cohort_id: str
    gender: str
    date: str
    subj_id: str
    subj_fold: Path
    cohort_fold: Path
    preprocessed: bool

def has_head(path: Path,treshold=200) -> bool:
    """
    return false if no head or head is small
    :param path: path of folder
    :return: True if folder have been processed
    """
    x = list(path.glob("**/*_head"))
    if len(x) > 0:
        head_folder = x[0]
        if len(list(head_folder.glob("**/*.bmp") )) > treshold:

            return True
        else:
            print("Failed coubt of head images", len(list(head_folder.glob("**/*.bmp") )))
            return False
    return False


class ImageProperties:
    def __init__(self, ini_file):
        config = ConfigParser()
        config.read(ini_file,encoding='cp1252')

        # System
        self.scanner = config.get('System', 'Scanner')
        self.instrument_sn = config.get('System', 'Instrument S/N')
        self.home_directory = config.get('System', 'Home Directory')
        self.source_type = config.get('System', 'Source Type')
        self.camera_pixel_size = float(config.get('System', 'Camera Pixel Size (um)'))

        # Acquisition
        self.data_directory = config.get('Acquisition', 'Data Directory')
        self.filename_prefix = config.get('Acquisition', 'Filename Prefix')
        self.number_of_files = int(config.get('Acquisition', 'Number Of Files'))
        self.number_of_rows = int(config.get('Acquisition', 'Number Of Rows'))
        self.number_of_columns = int(config.get('Acquisition', 'Number Of Columns'))
        self.filename_index_length = int(config.get('Acquisition', 'Filename Index Length'))
        #self.frame_index_length = int(config.get('Acquisition', 'Frame Index Length'))
        #self.partial_width = config.get('Acquisition', 'Partial Width')
        self.image_crop_origin_x = int(config.get('Acquisition', 'Image crop origin X'))
        self.image_crop_origin_y = int(config.get('Acquisition', 'Image crop origin Y'))
        self.camera_binning = config.get('Acquisition', 'Camera binning')
        self.image_rotation = float(config.get('Acquisition', 'Image Rotation'))
        self.optical_axis = int(config.get('Acquisition', 'Optical Axis (line)'))
        self.object_to_source = float(config.get('Acquisition', 'Object to Source (mm)'))
        self.camera_to_source = float(config.get('Acquisition', 'Camera to Source (mm)'))
        self.source_voltage = int(config.get('Acquisition', 'Source Voltage (kV)'))
        self.source_current = int(config.get('Acquisition', 'Source Current (uA)'))
        self.image_pixel_size = float(config.get('Acquisition', 'Image Pixel Size (um)'))
        #self.scaled_image_pixel_size = float(config.get('Acquisition', 'Scaled Image Pixel Size (um)'))
        self.image_format = config.get('Acquisition', 'Image Format')
        self.depth = int(config.get('Acquisition', 'Depth (bits)'))
        self.exposure = int(config.get('Acquisition', 'Exposure (ms)'))
        self.rotation_step = float(config.get('Acquisition', 'Rotation Step (deg)'))
        self.use_360_rotation = config.get('Acquisition', 'Use 360 Rotation')
        #self.scanning_position = float(config.get('Acquisition', 'Scanning position'))
        #self.frame_averaging = config.get('Acquisition', 'Frame Averaging')
        #self.flat_field_correction = config.get('Acquisition', 'Flat Field Correction')
        #self.geometrical_correction = config.get('Acquisition', 'Geometrical Correction')
        self.filter = config.get('Acquisition', 'Filter')



def convert_dataset_to_image(dataset: Dict[int,np.ndarray],image_props:ImageProperties):

    #check missing
    arr  = sorted(dataset)
    for i in range(len(sorted(arr))-1):
        if arr[i+1] - arr[i] > 1:
            print("Missing values between", arr[i], "and", arr[i+1])
            raise Exception()

    res_image = []
    for id in sorted(dataset):
        res_image.append( dataset[id] )

    res_array = np.array(res_image)
    affine_m = np.eye(4)
    affine_m[0,0] = affine_m[1,1] = affine_m[2,2] = image_props.image_pixel_size

    res_im = sitk.GetImageFromArray(res_array)
    voxel_size = (image_props.image_pixel_size/1000, image_props.image_pixel_size/1000,
                  image_props.image_pixel_size/1000)
    #res_im.SetDirection(affine_m.flatten())
    res_im.SetSpacing(voxel_size)
    return res_im


def convert_head_folder(fold : Union[str,Path]):
    """

    :param fold: folder with a head
    :return: Image
    """
    i_f = fold
    if isinstance(fold,str):
        i_f = Path(i_f)
    #folder already preprocessed
    rec_folder = [ x for x in i_f.iterdir() if x.is_dir() and x.name.endswith("Rec")][0]

    head_folder = list(rec_folder.glob("**/*_head"))[0]

    config_files = [x for x in head_folder.iterdir() if x.name.endswith(".log")]
    if len(config_files) == 0: # for manual files
        config_files = [x for x in head_folder.parent.iterdir() if x.name.endswith(".log")]
    config_file = config_files[0]
    config = ImageProperties(config_file)

    #image files
    dataset = {}
    for image_file in head_folder.glob("**/*.bmp"):
        #extra filter of head folder
        if image_file.name.endswith("spr.bmp"):
            continue
        ids = (int(image_file.name.replace(".bmp",'').split('rec')[-1]))
        arr = np.array(Image.open(image_file))
        dataset[ids] = arr

    return convert_dataset_to_image(dataset,config)


def process_cohort_folder():
    results = []
    not_cohort = ["BMD","TMD","Morphology","csv","+","workdir","checkpoints","_Screen","histogram"]
    for fold in PROCESS_DIR.iterdir():
        if not fold.is_dir():
            continue
        if any(substring in fold.name for substring in not_cohort):
            continue
        results.append(fold)
    return results


SRC_SUBJECTS = process_cohort_folder()
SRC_SUBJECTS = [fold.name for fold in SRC_SUBJECTS]
#make_output folder

OUT_FOLD = PROCESS_DIR / "skull_shape_workdir"
OUT_FOLD.mkdir(exist_ok=True)


rule all:
    input:
        (PROCESS_DIR/"logs_skull.txt")

rule make_nifti_image:
    input:
        PROCESS_DIR/"{subject}"
    output:
        FLAG = (OUT_FOLD/"{subject}/make_nifti_image.flag")
    run:

        #Path(output.FLAG).parent.mkdir(exist_ok=True)
        Path(output.FLAG).touch(exist_ok=True)
        if not has_head(Path(input[0]),200):
            print(f"Subject {input[0]} do not contain valid head")
            return
            #raise ValueError(f"Subject {input[0]} do not contain valid head")
        img = convert_head_folder(input[0])
        sitk.WriteImage(img,Path(output.FLAG).parent/"ct_image.nii.gz") # real output


def normalized_cross_correlation(image1, image2):
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)

    numerator = np.sum((image1 - mean1) * (image2 - mean2))
    denominator1 = np.sum((image1 - mean1) ** 2)
    denominator2 = np.sum((image2 - mean2) ** 2)

    ncc = numerator / (np.sqrt(denominator1 * denominator2))
    return ncc

def registration_quality( img1, img2):
    image1 = sitk.ReadImage(img1)
    image2 = sitk.ReadImage(img2)
    return normalized_cross_correlation(sitk.GetArrayFromImage(image1),sitk.GetArrayFromImage(image2))


def otsu_treshold(input_file: Path, out_file: Path):
    # Load image
    img = sitk.ReadImage(input_file)

    # Apply Otsu thresholding
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    otsu_img = otsu_filter.Execute(img)
    otsu_img.CopyInformation(img)
    # Save thresholded image
    sitk.WriteImage(otsu_img, out_file)


def flip_image(im_file_in: str, im_file_out: str, axis):
    """
    Flip an image along specified axes and preserve spatial alignment.

    Parameters:
        im_file_in (str): Path to the input image.
        im_file_out (str): Path to save the flipped image.
        axis (Tuple[bool, bool, bool]): Axes along which to flip (e.g., (True, False, False)).
    """
    if Path(im_file_out).exists():
        Path(im_file_out).unlink()

    # Read the input image
    image = sitk.ReadImage(im_file_in)

    # Flip the image
    flipped_image = sitk.Flip(image,axis)

    # Ensure spatial metadata (origin, spacing, direction) is consistent
    flipped_image.SetOrigin(image.GetOrigin())
    flipped_image.SetSpacing(image.GetSpacing())
    flipped_image.SetDirection(image.GetDirection())

    # Save the flipped image
    sitk.WriteImage(flipped_image,im_file_out)

def register_flip(floating_image: Union[str,Path],
                  reference_image: Union[str,Path], float_mask:Union[str,Path],
                  ref_mask :Union[str,Path],
                  outputs: Tuple[str, str],
                  inaff = None):
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
        cmd = [ 'reg_aladin','-ref',
                str(reference_image), '-rmask', ref_mask,
                '-flo', floating_image,'-fmask',float_mask,
                '-speeeed','-cog','-res',str(outputs[0]),'-aff', str(outputs[1])]
        cmd = "reg_aladin -ref '{refim}' -rmask '{rmask}' -flo '{flim}' -fmask '{fmask}'".format(refim=reference_image,
            rmask=ref_mask,flim=floating_image,fmask=float_mask)
        cmd += " -speeeeed -cog -res '" + str(outputs[0])
        cmd += "' -aff '" + outputs[1] + "'"
        print(cmd)
        shell(cmd)


def landmark_point_generation( input_file: str, output_file: str):
    net = PointNetRegressor(1000,9)
    net.load_state_dict(torch.load(control_points_model))
    net.eval()

    pts = net.compute_from_image(input_file)

    save_landmark_elastix(output_file,pts)

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
    image1 = sitk.ReadImage(image_path1, sitk.sitkFloat32)
    image2 = sitk.ReadImage(image_path2, sitk.sitkFloat32)


    # Convert images to numpy arrays
    image1_array = sitk.GetArrayFromImage(image1)
    image2_array = sitk.GetArrayFromImage(image2)

    # Compute the metric
    if metric == "mutual_information":
        joint_histogram, _, _ = np.histogram2d(image1_array.ravel(), image2_array.ravel(), bins=32)
        joint_prob = joint_histogram / np.sum(joint_histogram)
        marginal_prob1 = np.sum(joint_prob, axis=0)
        marginal_prob2 = np.sum(joint_prob, axis=1)
        mutual_information = np.sum(
            joint_prob[joint_prob > 0] * np.log(joint_prob[joint_prob > 0] /
                                                (marginal_prob1[np.newaxis, :] *
                                                 marginal_prob2[:, np.newaxis])[joint_prob > 0]))
        return mutual_information

    elif metric == "correlation":
        correlation = np.corrcoef(image1_array.ravel(), image2_array.ravel())[0, 1]
        return correlation

    else:
        raise ValueError(f"Unsupported metric: {metric}")


def perform_linear_coregistration(floating_image, reference_image, ref_landmarks_, ref_mask_, float_mask, outputs, inaff=None, use_landmarks=True):
    proces_folder = Path(floating_image).parent
    out_file = Path(outputs[0])
    # landmark generation
    landmark_file = proces_folder / "landmarks.pts"
    if use_landmarks:
        landmark_point_generation(ref_mask_, str(landmark_file))

        fix_pts_np = load_landmark_elastix(ref_landmarks_)
        mov_pts_np = load_landmark_elastix(landmark_file)
    # write alignment script

        aff_mat = compute_affine_transformation(mov_pts_np, fix_pts_np)
    else:
        aff_mat = np.eye(4)
    np.savetxt(proces_folder / "rigid_head.txt", aff_mat)
    update_image_affine(floating_image, output_head=out_file, affine_matrix=aff_mat)
    update_image_affine(float_mask
        ,proces_folder / "coregistered_mask.nii.gz", aff_mat)

    # linear coregistration
    register_flip(floating_image, reference_image, ref_mask=ref_mask_, float_mask=float_mask, outputs=outputs, inaff=inaff)
    metric = measure_coregistration_metric(outputs[0],reference_image,metric="correlation")
    return metric

def register_linear_full( floating_image: Union[str,Path],
                          reference_image: Union[str,Path],
                          float_mask:Union[str,Path],
                          ref_mask :Union[str,Path],
                          outputs: Tuple[str, str],
                          inaff = None):
    """

    :param floating_image:
    :param reference_image:
    :param float_mask:
    :param ref_mask:
    :param outputs:
    :param inaff:
    :return:
    """

    metrics_outputs =[]
    process_folder = Path(floating_image).parent
    metric_0 = perform_linear_coregistration(floating_image, reference_image, ref_landmarks, ref_mask, float_mask, outputs, inaff)
    metrics_outputs.append(metric_0)
    print("Metric",metric_0)

    flip_params = [(False,False,False),(False, True,True), (True,False, True), (True, True, False)]
    if metric_0 < 0.5:
        for fl_pam in flip_params:
            print("Flip",fl_pam)
            flip_image(floating_image,str(outputs[0]), axis=fl_pam)
            print("float mask",float_mask)
            flip_image(float_mask,str(process_folder/ "coregistered_mask.nii.gz"), axis=fl_pam)

            metric_1 = perform_linear_coregistration(outputs[0], reference_image, ref_landmarks, ref_mask, str(process_folder/ "coregistered_mask.nii.gz"), outputs, inaff,use_landmarks=False)
            print("Metric!!",metric_1)
            metrics_outputs.append(metric_1)
            if metric_1 > 0.8:
                break

    if len(metrics_outputs) == 1:
        return
    else:
        idx_max = np.argmax(metrics_outputs)
        print("Max metric",metrics_outputs[idx_max])
        print("idx_max",idx_max)
        flip_image(floating_image,outputs[0],axis=flip_params[idx_max])
        flip_image(float_mask,str(process_folder/ "coregistered_mask.nii.gz"),axis=flip_params[idx_max])
        if idx_max == 0:
            perform_linear_coregistration(outputs[0], reference_image, ref_landmarks, ref_mask, str(process_folder/ "coregistered_mask.nii.gz"), outputs, inaff,use_landmarks=True)
        else:
            perform_linear_coregistration(outputs[0], reference_image, ref_landmarks, ref_mask, str(process_folder/ "coregistered_mask.nii.gz"), outputs, inaff,use_landmarks=False)





rule otsu_threshold:
    input:
        (OUT_FOLD/"{subject}/make_nifti_image.flag")
    output:
        flag = (OUT_FOLD/"{subject}/1_otsu.flag")

    threads: 1

    run:


        inp_file =  Path(output.flag).parent / "ct_image.nii.gz"
        out_file = Path(output.flag).parent / "bone_coregistration_mask.nii.gz"
        if inp_file.exists():
            otsu_treshold(inp_file,out_file)
        Path(output.flag).touch(exist_ok=True)



# rule landmark_point_generation:
#     input:
#         OUT_FOLD/"{subject}/1_otsu.flag"
#     output:
#         OUT_FOLD/"{subject}/2_landmarks.flag"
#     params:
#         in_ct = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/bone_coregistration_mask.nii.gz",
#         out_landmarks = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/landmarks.pts"
#     threads: 1
#
#     run:
#         net = PointNetRegressor(1000,9)
#         net.load_state_dict(torch.load(control_points_model))
#         net.eval()
#         Path(output[0]).touch(exist_ok=True)
#         pts = net.compute_from_image(params.in_ct)
#
#         save_landmark_elastix(params.out_landmarks,pts)

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
    centroid_a = np.mean(a_points, axis=0)
    centroid_b = np.mean(b_points, axis=0)

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

def transform_points(points, T):
    points = np.array(points)
    num_points = points.shape[0]
    points_homogeneous = np.hstack([points, np.ones((num_points, 1))]).T  # Shape: (4, num_points)
    transformed_points = T @ points_homogeneous  # Shape: (4, num_points)
    transformed_points = transformed_points[:3].T  # Convert back to (num_points, 3)
    return transformed_points

def save_transform_to_tfm(matrix, filename):
    """
    Save a NumPy transformation matrix as a Slicer-readable .tfm file.

    Parameters:
    - matrix: A 4x4 NumPy array representing the homogeneous transformation matrix.
    - filename: The output filename (should end with .tfm).
    """
    import numpy as np

    # Ensure the matrix is a NumPy array
    matrix = np.array(matrix)
    # invert matrix to match the tfm format
    #matrix = np.linalg.inv(matrix)

    # Validate the matrix shape
    if matrix.shape != (4, 4):
        raise ValueError("The transformation matrix must be a 4x4 homogeneous matrix.")

    # Extract the rotation and translation components
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    # Flatten the rotation matrix and translation vector
    rotation_flat = rotation.flatten()
    translation_flat = translation.flatten()

    # Create the content of the .tfm file
    tfm_content = f"""#Insight Transform File V1.0
# Transform 0
Transform: AffineTransform_double_3_3
Parameters: {' '.join(map(str, rotation_flat))} {' '.join(map(str, translation_flat))}
FixedParameters: 0 0 0
"""

    # Write the content to the file
    with open(filename, 'w') as file:
        file.write(tfm_content)

    print(f"Transformation matrix saved to {filename}")


def parse_transform_parameters(file_content):
    """
    Parse the transformation parameters from a text input and return an affine transformation matrix.

    Args:
    - file_content (str): The content of the transformation file as a string.

    Returns:
    - affine_matrix (numpy.ndarray): A 4x4 numpy array representing the affine transformation matrix.
    """
    # Split the content into lines
    lines = file_content.splitlines()

    # Initialize empty variables to store the parameters
    transform_params = None

    # Loop through each line to find the TransformParameters and extract them
    for line in lines:
        if line.startswith("(TransformParameters"):
            # Extract the numerical values from the line (ignoring the keyword)
            params_str = line.replace("(TransformParameters", "").replace(")", "").strip()
            transform_params = list(map(float, params_str.split()))
            break  # We found the transform parameters, so we can stop searching

    if transform_params is None:
        raise ValueError("No TransformParameters found in the input text.")

    # Convert the transform_params to an affine matrix
    affine_matrix = convert_to_affine_matrix(transform_params)

    return affine_matrix


def convert_to_affine_matrix(transform_params):
    """
    Convert the transformation parameters from the configuration into a 4x4 affine transformation matrix.

    Args:
    - transform_params (list): List of 12 transformation parameters (9 for rotation/scale, 3 for translation).

    Returns:
    - affine_matrix (numpy.ndarray): A 4x4 numpy array representing the affine transformation matrix.
    """
    # Extract the 3x3 rotation/scale matrix from the first 9 elements of transform_params
    rotation_scale_matrix = np.array(transform_params[:9]).reshape(3,3)

    # Extract the translation vector from the last 3 elements of transform_params
    translation_vector = np.array(transform_params[9:12]).reshape(3,1)

    # Create a 4x4 identity matrix
    affine_matrix = np.eye(4)

    # Insert the 3x3 rotation/scale matrix into the top-left of the 4x4 affine matrix
    affine_matrix[:3, :3] = rotation_scale_matrix

    # Insert the translation vector into the last column of the affine matrix
    affine_matrix[:3, 3] = translation_vector.flatten()

    return affine_matrix

def update_image_affine(input_ct, output_head, affine_matrix):
    image_f = nib.load(input_ct)
    image_affine = image_f.affine
    new_affine = affine_matrix @ image_affine

    # Update the image's affine matrix
    new_image = nib.Nifti1Image(image_f.get_fdata(), new_affine, image_f.header)
    nib.save(new_image, output_head)

rule linear_coregistration:
    input: OUT_FOLD/"{subject}/1_otsu.flag"
    output: OUT_FOLD/"{subject}/2_lin_cor.flag"
    params:
        in_ct = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/ct_image.nii.gz",
        in_landmarks = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/landmarks.pts",
        in_mask = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/bone_coregistration_mask.nii.gz",
        out_head = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/coregistered_head.nii.gz",
        out_mat = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/rigid_head.txt",
        out_affine = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/affine_head.txt"

    threads: 2
    run:
        process_folder = Path(output[0]).parent
        register_linear_full(str(params.in_ct),str(ref_im),str(params.in_mask),ref_mask=str(ref_mask),

            outputs=(str(params.out_head), str(process_folder / 'TransformParameters.0.txt')))

        aff_mat = np.loadtxt(process_folder / 'rigid_head.txt')
        # read the affine matrix from the file
        affine_matrix = np.loadtxt(process_folder / 'TransformParameters.0.txt')

        full_head_matrix = affine_matrix @ aff_mat

        np.savetxt(str(params.out_affine),full_head_matrix)
        #print("difference after affine transform",fix_pts_np - transform_points(mov_pts_np,full_head_matrix))
        #

        fn1 = process_folder / 'TransformParameters.0.txt'
        shell(f'rm -f "{str(fn1)}"')

        #raise ValueError("STOP")
        Path(output[0]).touch(exist_ok=True)

def non_rigid_registration(flo_im: str, ref_im: str, out_transform: str, out_image: str, param_file:str):
    # Reading
    fold = Path(out_image).parent
    cmd = f"elastix -f '{ref_im}' -m '{flo_im}' -p '{param_file}' -out '{str(fold)}'"
    shell(cmd)
    shell(f"mv -f '{str(fold/ 'result.0.nii.gz')}' '{out_image}'")
    shell(f"mv -f '{str(fold/ 'TransformParameters.0.txt')}' '{out_transform}'")

rule non_rigid_registration_rule:
    input: (OUT_FOLD/"{subject}/2_lin_cor.flag")
    output: (OUT_FOLD/"{subject}/3_nonrig.flag")
    params:
        out_nlim = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/nonlin_image.nii.gz",
        out_txt = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/nl_transformation_params.txt",
        inp_coregim = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/coregistered_head.nii.gz"
    threads: 2
    run:
        Path(output[0]).touch(exist_ok=True)
        if not params.inp_coregim.exists():
            return
        # script coregisters reference image (choosen as default to each linearly registered head for reverse mesh generation
        or_ref_im = REF_IMAGE / 'oriented_image.nii.gz'
        non_rigid_registration(flo_im=str(or_ref_im), ref_im=str(params.inp_coregim),
            out_transform=str(params.out_txt),out_image=str(params.out_nlim),param_file=str(PARAMETER_FILE_NONLIN))


rule inverse_transform_mesh_rule:
    input: (OUT_FOLD/"{subject}/3_nonrig.flag")
    output: (OUT_FOLD/"{subject}/4_inv_mesh.flag")
    params:
        in_ct = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/nl_transformation_params.txt",
        out_head = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/head_affine_registered.obj",

    threads: 2
    run:
        dir1name = str(Path(input[0]).parent)
        Path(output[0]).touch(exist_ok=True)
        #generate deformation field
        if not Path(params.in_ct).exists():
            return
        cmd = f"transformix -tp '{params.in_ct}' -out '{dir1name}' -def all"
        shell (cmd)
        or_ref_mesh = REF_IMAGE / "oriented_mesh.obj"
        print(or_ref_mesh)

        def_field_file  = dir1name + os.sep + "deformationField.nii.gz"
        ###deform original mesh

        cmd = (f"xvfb-run -a  {SLICER_CORE} --python-script {MESH_TRANSFORM_SCRIPT} '{def_field_file}' '{or_ref_mesh}' "
               f"'{params.out_head}'  --testing --no-splash --no-main-window")
        print(cmd)
        shell(cmd)
        shell("rm -rf '" + def_field_file +"'" )





### rule for computing of mahalanobis pic's results

def read_coh_file():
    coh_file_l = open(coh_file,'rb')
    COHORTS = pickle.load(coh_file_l)

    coh_file_l.close()
    return COHORTS
    print("FILE READ")


def sutures_open(image_file,mesh_file):
    model = SuturesNetWrapper(config['sutures_model'])
    return model.predict(image_file,mesh_file)


rule general_procrustes:
    input: (OUT_FOLD/"{subject}/6_analyse_scalling.flag")
    output: (OUT_FOLD/"{subject}/6_procr_result.flag")
    params:
        in_ct = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/mahalanobis.mat",
        in_image = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/coregistered_head.nii.gz",
        in_matr = lambda wildcards: OUT_FOLD/ f"{wildcards.subject}/result_head.mat",
        head = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/head_affine_registered.obj",
        out_file = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/abnormality_automated.csv",
    run:
        Path(output[0]).touch(exist_ok=True)

        if Path(params.head).exists():
            read_orig_mesh = read_vtk_obj(str(params.head))
            ref_mesh_pts = get_mesh_vertices(read_orig_mesh) - np.array(center_of_mesh(read_orig_mesh))

            # read normal points
            cohorts = read_coh_file()
            subject_gender = Path(output[0]).parent.name.split("_")[1]
            ref_mesh,_,_ = gpa(cohorts[subject_gender])

            _,distance,_ = procreustes_alignment(ref_mesh,ref_mesh_pts)


            #load 99 percentiles and compute percentage
            face_perc_file = MALE_FACE_PERC if subject_gender == 'M' else FEMALE_FACE_PERC

            face_perc_data = scio.loadmat(face_perc_file)
            perc99 = face_perc_data['perc99']
            energies =  scio.loadmat(str(params.in_ct))['energies']
            ratio = (energies > perc99).sum() / perc99.shape[1]
            is_suture = sutures_open(params.in_image,params.head)

            ##
            mat = np.loadtxt(params.in_matr)
            u,s,v = np.linalg.svd(mat)

            array = {"proc_dist": distance,
                     "abn_distance": "normal" if distance<980 else "abnormal", "sutures_open" : is_suture,
                     "area99perc" : ratio, "abn_perc" : "normal" if ratio<0.1 else "abnormal","s_x": s[0],"s_y": s[1],"s_z":s[2]}
            df = pd.Series(array)
            df.to_csv(params.out_file)




rule combine_results:
    input: expand(OUT_FOLD/"{subject}/7_gifs.flag",subject = SRC_SUBJECTS)
    output: (PROCESS_DIR/"abnormality_automated_combined.csv")
    params:
        out_file = OUT_FOLD/"abnormality_automated_combined.csv"
    run:
        columns = ["proc_dist", "abn_distance",
                   "area99perc", "abn_perc", 'sutures_open', "s_x", "s_y", "s_z"]
        index = []
        res = []
        for fold in input:
            p_fold = Path(fold).parent
            index.append(p_fold.name)
            if (p_fold/ "abnormality_automated.csv").exists():
                subject =pd.read_csv((p_fold/ "abnormality_automated.csv"),header = 0,index_col='Unnamed: 0')
                subject = subject.transpose().iloc[0]

                res.append(dict(subject))
            else:
                res.append({x:None for x in columns})
        print(res)
        pd.DataFrame(res,index=index).to_csv(output[0])

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


def create_mah_gif(input_mat_file, input_obj_file, output_gif_file, title ):
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

    fig.colorbar(m, ax=ax)

    cmap = plt.get_cmap(cmp_str)
    norm = plt.Normalize(0, 100)
    cmap = plt.get_cmap(cmp_str)
    p3dc= ax.plot_trisurf(pts[:, 0],pts[:, 1],pts[:, 2],
        triangles=faces,linewidth=0.2  #,)
    )
    ax.set_box_aspect([np.ptp(i) for i in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    p3dc.set_fc(cmap(norm(energies)))
    #ax.set_autoscale_on(False)
    # plt.savefig(output_gif_file.split(".")[0] + ".png")
    # Create the animation function
    def animate(frame):

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
        ax.set_title(title)

        # Rotate the view
        ax.view_init(elev=20,azim=frame * 36)  # Adjust the rotation speed (36 degrees per frame)

    # Number of frames in the GIF
    num_frames = 10

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=num_frames)
    print("Attempting to save gif to ", output_gif_file)
    # Save the animation as a GIF
    ani.save(output_gif_file, writer='pillow',fps=1 )

    plt.close(fig)
    print('GIF saved successfully.')

def create_perc_gif(input_mat_file, input_obj_file, output_fig_file, face_perc_file, title=None ):
    # Load the energies
    energies = scio.loadmat(input_mat_file)['energies']
    #energies = energies.T  # Transpose if needed

    # Load the mesh
    pts, faces, mesh = read_mesh_and_points(input_obj_file)

    # Load face vertices from face_perc_file (assuming it's a numpy file)
    face_perc_data = scio.loadmat(face_perc_file)
    perc75 = face_perc_data['perc75']
    perc95 = face_perc_data['perc95']
    perc99 = face_perc_data['perc99']

    # Create colormap based on energies
    colormap_values = np.ones_like(energies)
    colormap_values[energies > perc75] = 2  # Green
    colormap_values[energies > perc95] = 3  # Red
    colormap_values[energies > perc99] = 4  # Red

    cmp = []
    print(colormap_values.shape)
    for i in range(colormap_values.shape[1]):
        if colormap_values[0,i]==1:
            cmp.append('blue')
        elif colormap_values[0,i]== 2:
            cmp.append('green')
        elif colormap_values[0,i]== 3:
            cmp.append('yellow')
        elif colormap_values[0,i]== 4:
            cmp.append('red')
    # for x in colormap_values.tolist():
    #     if x == 1:
    #         cmp.append([0, 0, 1, 1])  # Blue
    #     elif x == 2:
    #         cmp.append([0, 1, 0, 1])  # Green
    #     elif x == 3:
    #         cmp.append([1, 1, 0, 1])  # Yellow
    #     elif x == 4:
    #         cmp.append([1, 0, 0, 1])  # Red

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the colormap with RGBA values

    p3dc = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                           triangles=faces, linewidth=0.20)

    # Ensure the number of face colors matches the number of faces
    if len(cmp) != len(faces):
        print("Number of face colors doesn't match number of faces." , len(cmp), len(faces))
    # else:
    p3dc.set_fc(cmp)
    #p3dc.set_fc(cmp)

    # cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap))
    # cbar.set_ticks([1, 2, 3, 4])
    # cbar.set_ticklabels(['< 75%', '> 75%', '> 95%', '> 99%'])

    ax.set_box_aspect([np.ptp(i) for i in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    plt.savefig(str(output_fig_file).split(".")[0] + ".png")

    def animate(frame):

        ax.set_box_aspect([np.ptp(i) for i in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        # Rotate the view
        ax.view_init(elev=20,azim=frame * 36)  # Adjust the rotation speed (36 degrees per frame)

    # Number of frames in the GIF
    num_frames = 10

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=num_frames)

    # Save the animation as a GIF
    ani.save(output_fig_file, writer='pillow',fps=1 )

    plt.close(fig)
    print('GIF saved successfully.')

# Example usage:
# create_perc_fig('input_energies.npy', 'input_mesh.obj', 'output_figure.png', 'face_perc_data.npy')

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

rule compute_mahalanobis_heads:
    input: (OUT_FOLD/"{subject}/4_inv_mesh.flag")
    output: (OUT_FOLD/"{subject}/5_mah_mat.flag")
    params:
        in_ct = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/nl_transformation_params.txt",
        head = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/head_affine_registered.obj",
        out_file = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/mahalanobis.mat",
    threads: 1
    run:
        COHORTS = read_coh_file()
        par_name = Path(input[0]).parent.name
        Path(output[0]).touch(exist_ok=True)
        if Path(params.head).exists():
            cohort = COHORTS[par_name.split("_")[1]] # copy males or females

            mesh_pts,faces, mesh = read_mesh_and_points(params.head)


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
            scio.savemat(params.out_file,{"energies": energies})


rule mah_gifs:
    input: OUT_FOLD/"{subject}/6_procr_result.flag"
    output: (OUT_FOLD/"{subject}/7_gifs.flag")
    params:
        in_mat = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/mahalanobis.mat",
        head = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/head_affine_registered.obj",
        out_mah_file = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/mahalanobis.gif",
        out_perc_file= lambda wildcards: OUT_FOLD/ f"{wildcards.subject}/perc.gif",
    run:
        Path(output[0]).touch(exist_ok=True)
        gender = Path(input[0]).parent.name.split("_")[1]
        if gender == "M":
            fp_f = MALE_FACE_PERC
        else:
            fp_f = FEMALE_FACE_PERC
        if Path(params.head).exists():
            create_perc_gif(input_mat_file=params.in_mat,
                input_obj_file=params.head,
                output_fig_file=params.out_perc_file,face_perc_file=fp_f,title=Path(input[0]).parent.name)

            create_mah_gif(input_mat_file=params.in_mat,
                input_obj_file=params.head,
                output_gif_file=str(params.out_mah_file),title=Path(input[0]).parent.name)
            

rule result_rotation_scalling:
    input: OUT_FOLD/"{subject}/5_mah_mat.flag"
    output: (OUT_FOLD/"{subject}/6_analyse_scalling.flag")
    params:
        in_mat1 = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/mahalanobis.mat",
        head = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/head_affine_registered.obj",
        in_mat = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/affine_head.txt",
        out_mat = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/result_head.mat"
    threads: 1
    run:
        Path(output[0]).touch(exist_ok=True)
        if Path(params.in_mat).exists():
            or_ref_ = REF_IMAGE / "oriented_mesh.obj"
            read_orig_mesh = read_vtk_obj(str(or_ref_))
            ref_mesh_pts= get_mesh_vertices(read_orig_mesh) - np.array(center_of_mesh(read_orig_mesh))
            input_mesh = read_vtk_obj(params.head)
            input_mesh_pts= get_mesh_vertices(input_mesh) - np.array(center_of_mesh(input_mesh))

            _,_,matrix = procreustes_alignment(ref_mesh_pts,input_mesh_pts)

            or_ref_im = REF_IMAGE / "oriented_image.nii.gz"
            print("outmat",matrix)
            prev_mat = np.loadtxt(params.in_mat)
            prev_mat = prev_mat[:3,:3]

            print(prev_mat)
            out_mat = np.dot( matrix, prev_mat)

            np.savetxt(params.out_mat,out_mat)




            
def analyse_dirs_logs(folder: Union[str,Path]):
    text = " "
    folder = Path(folder)
    if not (folder / "ct_image.nii.gz").exists():
        text += "Head was not found. Check original subject"
    elif not (folder / "affine_head.mat").exists():
        text += "Linear coregistration failed. Check log of job and check original subject"
    elif not (folder / "bone_coregistration_mask.nii.gz").exists():
        text += "Mask generation failed. Check log of job and check original subject"
    elif not (folder / "bone_coregistration_mask.nii.gz").exists():
        text += "Mask generation failed. Check log of job and check original subject"
    elif not (folder / "nonlin_image.nii.gz").exists():
        text += "Non linear coregistration. Check log of job and ct image"
    elif not (folder / "head_affine_registered.obj").exists():
        text += "Conversion of meshes failed. Check log of job and coregistration results"
    elif not (folder / "mahalanobis.mat").exists():
        text += "Mahalanobis computation problems. Check log "
    elif not (folder / "abnormality_automated.csv").exists():
        text += "Abnormality prining failed. Check original logs"
        return text
    elif not (folder / "mahalanobis.gif").exists():
        text += "GIF saving failed. Check original logs"
    return text + "\n"


rule log_generation:
    input:
        (PROCESS_DIR/"abnormality_automated_combined.csv")
    output:
        (PROCESS_DIR / "logs_skull.txt")
    run:
        logstr = ""
        for x in SRC_SUBJECTS:

            fold = OUT_FOLD / x
            logstr += "Subject "+ str(x) + ": " + str(analyse_dirs_logs(fold))
        Path(output[0]).touch(exist_ok=True)
        Path(output[0]).write_text(logstr)

