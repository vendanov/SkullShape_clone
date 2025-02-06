import os
import pickle
import shlex
import subprocess
from builtins import all

import PIL
import torchvision.transforms as T
import xml.etree.ElementTree as ET

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
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.v2.functional import affine
from tqdm.auto import tqdm

from mouse_skull_pipelines.scripts.gpa import gpa, procreustes_alignment
from mouse_skull_pipelines.scripts.image_algorithms import create_oriented_bounding_box, marching_cubes, \
    crop_image_with_obb
from mouse_skull_pipelines.scripts.mesh_algorithms import save_landmark_elastix, load_landmark_elastix
from mouse_skull_pipelines.scripts.nets import PointNetRegressor, SuturesNetWrapper, coalign_roi_based
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
OTSU_REMOVE_OUTER = config['otsu_remove_outer']
LINEAR_SIMPLE = config['linreg_simple']


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


def has_head(path: Path, treshold=200) -> bool:
    """
    return false if no head or head is small
    :param path: path of folder
    :return: True if folder have been processed
    """
    x = list(path.glob("**/*_head"))
    if len(x) > 0:
        head_folder = x[0]
        if len(list(head_folder.glob("**/*.bmp"))) > treshold:

            return True
        else:
            print("Failed coubt of head images",len(list(head_folder.glob("**/*.bmp"))))
            return False
    return False


class ImageProperties:
    def __init__(self, ini_file):
        config = ConfigParser()
        config.read(ini_file,encoding='cp1252')

        # System
        self.scanner = config.get('System','Scanner')
        self.instrument_sn = config.get('System','Instrument S/N')
        self.home_directory = config.get('System','Home Directory')
        self.source_type = config.get('System','Source Type')
        self.camera_pixel_size = float(config.get('System','Camera Pixel Size (um)'))

        # Acquisition
        self.data_directory = config.get('Acquisition','Data Directory')
        self.filename_prefix = config.get('Acquisition','Filename Prefix')
        self.number_of_files = int(config.get('Acquisition','Number Of Files'))
        self.number_of_rows = int(config.get('Acquisition','Number Of Rows'))
        self.number_of_columns = int(config.get('Acquisition','Number Of Columns'))
        self.filename_index_length = int(config.get('Acquisition','Filename Index Length'))
        #self.frame_index_length = int(config.get('Acquisition', 'Frame Index Length'))
        #self.partial_width = config.get('Acquisition', 'Partial Width')
        self.image_crop_origin_x = int(config.get('Acquisition','Image crop origin X'))
        self.image_crop_origin_y = int(config.get('Acquisition','Image crop origin Y'))
        self.camera_binning = config.get('Acquisition','Camera binning')
        self.image_rotation = float(config.get('Acquisition','Image Rotation'))
        self.optical_axis = int(config.get('Acquisition','Optical Axis (line)'))
        self.object_to_source = float(config.get('Acquisition','Object to Source (mm)'))
        self.camera_to_source = float(config.get('Acquisition','Camera to Source (mm)'))
        self.source_voltage = int(config.get('Acquisition','Source Voltage (kV)'))
        self.source_current = int(config.get('Acquisition','Source Current (uA)'))
        self.image_pixel_size = float(config.get('Acquisition','Image Pixel Size (um)'))
        #self.scaled_image_pixel_size = float(config.get('Acquisition', 'Scaled Image Pixel Size (um)'))
        self.image_format = config.get('Acquisition','Image Format')
        self.depth = int(config.get('Acquisition','Depth (bits)'))
        self.exposure = int(config.get('Acquisition','Exposure (ms)'))
        self.rotation_step = float(config.get('Acquisition','Rotation Step (deg)'))
        self.use_360_rotation = config.get('Acquisition','Use 360 Rotation')
        #self.scanning_position = float(config.get('Acquisition', 'Scanning position'))
        #self.frame_averaging = config.get('Acquisition', 'Frame Averaging')
        #self.flat_field_correction = config.get('Acquisition', 'Flat Field Correction')
        #self.geometrical_correction = config.get('Acquisition', 'Geometrical Correction')
        self.filter = config.get('Acquisition','Filter')


def convert_dataset_to_image(dataset: Dict[int, np.ndarray], image_props: ImageProperties):
    #check missing
    arr = sorted(dataset)
    for i in range(len(sorted(arr)) - 1):
        if arr[i + 1] - arr[i] > 1:
            print("Missing values between",arr[i],"and",arr[i + 1])
            raise Exception()

    res_image = []
    for id in sorted(dataset):
        res_image.append(dataset[id])

    res_array = np.array(res_image)
    affine_m = np.eye(4)
    affine_m[0, 0] = affine_m[1, 1] = affine_m[2, 2] = image_props.image_pixel_size

    res_im = sitk.GetImageFromArray(res_array)
    voxel_size = (image_props.image_pixel_size / 1000, image_props.image_pixel_size / 1000,
                  image_props.image_pixel_size / 1000)
    #res_im.SetDirection(affine_m.flatten())
    res_im.SetSpacing(voxel_size)
    return res_im


def convert_head_folder(fold: Union[str, Path]):
    """

    :param fold: folder with a head
    :return: Image
    """
    i_f = fold
    if isinstance(fold,str):
        i_f = Path(i_f)
    #folder already preprocessed
    rec_folder = [x for x in i_f.iterdir() if x.is_dir() and x.name.endswith("Rec")][0]

    head_folder = list(rec_folder.glob("**/*_head"))[0]

    config_files = [x for x in head_folder.iterdir() if x.name.endswith(".log")]
    if len(config_files) == 0:  # for manual files
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

    return convert_dataset_to_image(dataset,config), config


def process_cohort_folder():
    results = []
    not_cohort = ["BMD", "TMD", "Morphology", "csv", "+", "workdir", "checkpoints", "_Screen", "histogram"]
    for fold in PROCESS_DIR.iterdir():
        if not fold.is_dir():
            continue
        if any(substring in fold.name for substring in not_cohort) or fold.name.endswith("-") or fold.name.endswith(
                "!"):
            continue
        results.append(fold)
    return results


SRC_SUBJECTS = process_cohort_folder()
SRC_SUBJECTS = [fold.name for fold in SRC_SUBJECTS]
#make_output folder

OUT_FOLD = PROCESS_DIR / "skull_shape_workdir"
OUT_FOLD.mkdir(exist_ok=True)


def is_rigid_transform(matrix):
    """
    Checks if the given 4x4 matrix represents a rigid transform (no scaling or shearing).

    Args:
        matrix (np.ndarray): 4x4 transformation matrix.

    Returns:
        bool: True if the matrix is a rigid transform, False otherwise.
    """
    rotation_matrix = matrix[:3, :3]

    # Check orthonormality
    is_orthonormal = np.allclose(np.dot(rotation_matrix,rotation_matrix.T),np.eye(3),atol=1e-6)

    # Check determinant (should be 1 for rigid transforms)
    is_det_one = np.isclose(np.linalg.det(rotation_matrix),1.0,atol=1e-6)

    return is_orthonormal and is_det_one


rule all:
    input:
        (PROCESS_DIR / "logs_skull.txt")

    # expand(OUT_FOLD / "{subject}" / '2_lin_cor.flag',subject=SRC_SUBJECTS)


def extract_labels(xml_file: str):
    """
    Extracts all unique class labels from a Pascal VOC XML file.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        Dict: boxes and labels.
    """

    # Parse XML annotation
    boxes = []
    labels = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall("object"):
        class_name = obj.find("name").text

        labels.append(class_name)
        bbox = obj.find("bndbox")
        xmin = round(float(bbox.find("xmin").text))
        ymin = round(float(bbox.find("ymin").text))
        xmax = round(float(bbox.find("xmax").text))
        ymax = round(float(bbox.find("ymax").text))
        boxes.append([xmin, ymin, xmax, ymax])

    target = {"boxes": boxes, "labels": labels}
    return target


rule make_nifti_image:
    input:
        PROCESS_DIR / "{subject}"
    output:
        FLAG=(OUT_FOLD / "{subject}/make_nifti_image.flag")
    run:

        #Path(output.FLAG).parent.mkdir(exist_ok=True)
        Path(output.FLAG).touch(exist_ok=True)

        subj_name = Path(output.FLAG).parent.name

        if (PROCESS_DIR / 'full_body_workdir' / subj_name / 'full_body_top_classes.xml').exists():
            print(f"working with full_boudy {subj_name}")

            target = extract_labels(str(PROCESS_DIR / 'full_body_workdir' / subj_name / 'full_body_top_classes.xml'))
            print(target)

            # get head bounding box

            for i in range(len(target['labels'])):
                if target['labels'][i] == 'head':
                    head_bbox = target['boxes'][i]
                    break

            # read the image
            img = nib.load(str(PROCESS_DIR / 'full_body_workdir' / subj_name / 'full_body_image_raw.nii.gz'))

            affine = img.affine

            for i in range(3):
                if affine[i, i] < 0:
                    affine[i, i] = - affine[i, i]

            # get the head image
            img_fdata = img.get_fdata()

            head_img = img_fdata[int(head_bbox[0]):int(head_bbox[2]), :, int(head_bbox[1]):int(head_bbox[3])]
            head_img = head_img.astype(np.uint8)
            # save the head image
            head_img_nifti = nib.Nifti1Image(head_img,affine)
            nib.save(head_img_nifti,str(OUT_FOLD / subj_name / 'ct_image.nii.gz'))

            return

        if not has_head(Path(input[0]),200):
            print(f"Subject {input[0]} do not contain valid head")
            return
        #raise ValueError(f"Subject {input[0]} do not contain valid head")
        img, cfg = convert_head_folder(input[0])

        if '1176' in cfg.scanner:
            print("Flipping image")
            img_data = sitk.GetArrayFromImage(img)

            flipped_image = np.flip(img_data,axis=(1, 2))
            #flipped_image = img_data
            flipped_image = sitk.GetImageFromArray(flipped_image)
            #flipped_image.SetOrigin(img.GetOrigin())
            flipped_image.SetSpacing(img.GetSpacing())
            img = flipped_image
        #flipped_image.SetDirection(img.GetDirection())

        sitk.WriteImage(img,Path(output.FLAG).parent / "ct_image.nii.gz")  # real output


#sys.exit(-1)


def normalized_cross_correlation(image1, image2):
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)

    numerator = np.sum((image1 - mean1) * (image2 - mean2))
    denominator1 = np.sum((image1 - mean1) ** 2)
    denominator2 = np.sum((image2 - mean2) ** 2)

    ncc = numerator / (np.sqrt(denominator1 * denominator2))
    return ncc


def registration_quality(img1, img2):
    image1 = sitk.ReadImage(img1)
    image2 = sitk.ReadImage(img2)
    return normalized_cross_correlation(sitk.GetArrayFromImage(image1),sitk.GetArrayFromImage(image2))


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


def flip_image(im_file_in: str, im_file_out: str, axis, output_format=sitk.sitkInt8):
    if Path(im_file_out).exists():
        Path(im_file_out).unlink()
    image = sitk.ReadImage(im_file_in,output_format)

    array = sitk.GetArrayFromImage(image)

    axis_test = res = [x for x in range(len(axis)) if axis[x]]

    if axis == (False, False, False):
        flipped_array = array
        print("No flip")
    else:
        flipped_array = np.flip(array,axis=axis_test)

    new_img = sitk.GetImageFromArray(flipped_array)
    new_img.SetOrigin(image.GetOrigin())
    new_img.SetSpacing(image.GetSpacing())
    new_img.SetDirection(image.GetDirection())

    sitk.WriteImage(new_img,im_file_out,useCompression=True)


# def flip_image(im_file_in: str, im_file_out: str, axis):
#     """
#     Flip an image along specified axes and preserve spatial alignment.
#
#     Parameters:
#         im_file_in (str): Path to the input image.
#         im_file_out (str): Path to save the flipped image.
#         axis (Tuple[bool, bool, bool]): Axes along which to flip (e.g., (True, False, False)).
#     """
#     if Path(im_file_out).exists():
#         Path(im_file_out).unlink()
#
#     # Read the input image
#     image = sitk.ReadImage(im_file_in)
#
#     # Flip the image
#     flipped_image = sitk.Flip(image,axis)
#
#     # Ensure spatial metadata (origin, spacing, direction) is consistent
#     flipped_image.SetOrigin(image.GetOrigin())
#     flipped_image.SetSpacing(image.GetSpacing())
#     flipped_image.SetDirection(image.GetDirection())
#
#     # Save the flipped image
#     sitk.WriteImage(flipped_image,im_file_out,useCompression=True)

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
        cmd = ['reg_aladin', '-ref',
               str(reference_image), '-rmask', ref_mask,
               '-flo', floating_image, '-fmask', float_mask,
               '-speeeed', '-cog', '-res', str(outputs[0]), '-aff', str(outputs[1])]
        cmd = "reg_aladin -ref '{refim}' -rmask '{rmask}' -flo '{flim}' -fmask '{fmask}'".format(refim=reference_image,
            rmask=ref_mask,flim=floating_image,fmask=float_mask)
        cmd += " -speeeeed -cog -res '" + str(outputs[0])
        cmd += "' -aff '" + outputs[1] + "'"
        print(cmd)
        shell(cmd)


def landmark_point_generation(input_file: str, output_file: str):
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


def perform_linear_coregistration(floating_image, reference_image, ref_landmarks_, ref_mask_, float_mask, outputs, inaff=None, use_landmarks=True):
    proces_folder = Path(floating_image).parent
    out_file = Path(outputs[0])
    # landmark generation
    landmark_file = proces_folder / "landmarks.pts"
    if use_landmarks:
        landmark_point_generation(ref_mask_,str(landmark_file))

        fix_pts_np = load_landmark_elastix(ref_landmarks_)
        mov_pts_np = load_landmark_elastix(landmark_file)
        # write alignment script

        aff_mat = compute_affine_transformation(mov_pts_np,fix_pts_np)
    else:
        aff_mat = np.eye(4)
    np.savetxt(proces_folder / "rigid_head.txt",aff_mat)
    update_image_affine(floating_image,output_head=out_file,affine_matrix=aff_mat)
    update_image_affine(float_mask,output_head=proces_folder / "coregistered_mask.nii.gz",
        affine_matrix=aff_mat,copy=False,copy_im=floating_image)
    if Path(float_mask).name == "coregistered_mask.nii.gz":
        print("Mask is already coregistered")
    #sys.exit(-2)
    # linear coregistration
    register_flip(floating_image,reference_image,ref_mask=ref_mask_,float_mask=float_mask,outputs=outputs,inaff=inaff)
    metric = measure_coregistration_metric(outputs[0],reference_image,metric="correlation")
    return metric


def register_ants(flo_im, flo_mask, ref_im):
    import ants
    flo = ants.image_read(flo_im)
    ref = ants.image_read(ref_im)
    flo_msk = ants.image_read(flo_mask)

    out = ants.registration(fixed=ref,moving=flo,type_of_transform='Affine',aff_metric="meansquares",verbose=True)
    out_mat = out['fwdtransforms'][0]

    # read_out_mat
    #out_mat = ants.read_transform(out_mat)
    # convert to numpy array
    #write images
    Path(flo_im.replace(".nii.gz","_reg.nii.gz")).unlink(missing_ok=True)
    Path(flo_mask.replace(".nii.gz","_reg.nii.gz")).unlink(missing_ok=True)

    flo_reg = ants.apply_transforms(fixed=ref,moving=flo,transformlist=out['fwdtransforms'])
    flo_reg.to_file(flo_im.replace(".nii.gz","_reg.nii.gz"))

    # copy file


    flo_mask_reg = ants.apply_transforms(fixed=ref,moving=flo_msk,transformlist=out['fwdtransforms'])
    flo_mask_reg.to_file(flo_mask.replace(".nii.gz","_reg.nii.gz"))
    shell(f"cp -f '{flo_im.replace('.nii.gz','_reg.nii.gz')}' '{str(flo_im)}'")
    shell(f"cp -f '{flo_mask.replace('.nii.gz','_reg.nii.gz')}' '{str(flo_mask)}'")


def generate_projection_image(input_image: sitk.Image, axis=1):
    # find the first image
    projection_filter = sitk.MaximumProjectionImageFilter()
    projection_filter.SetProjectionDimension(axis)
    projection_image = projection_filter.Execute(input_image)
    return projection_image


def change_orientation(refim_file, flo_im_file, flo_mask_file=None):
    fixed_image = sitk.ReadImage(refim_file,sitk.sitkFloat32)
    moving_image = sitk.ReadImage(flo_im_file,sitk.sitkFloat32)

    shrink_factor = [4, 4, 4]
    fixed_image_downsampled = sitk.Shrink(fixed_image,shrink_factor)
    moving_image_downsampled = sitk.Shrink(moving_image,shrink_factor)

    registration_method = generate_orienter(fixed_image,moving_image)

    # Connect all of the observers so that we can perform plotting during registration.

    final_transform = registration_method.Execute(fixed_image_downsampled,moving_image_downsampled)

    print("matrix weew",get_transformation_matrix(final_transform),is_rigid_transform(get_transformation_matrix(final_transform)))

    # im  resample
    moving_resampled = sitk.Resample(moving_image,fixed_image,final_transform,sitk.sitkNearestNeighbor,0.0,moving_image.GetPixelID())
    os.remove(flo_im_file)
    sitk.WriteImage(moving_resampled,flo_im_file)

    # mask resample
    if flo_mask_file is not None:
        flo_mask = sitk.ReadImage(flo_mask_file,sitk.sitkUInt8)

        moving_resampled = sitk.Resample(flo_mask,fixed_image,final_transform,sitk.sitkNearestNeighbor,0.0,flo_mask.GetPixelID())
        os.remove(flo_mask_file)
        sitk.WriteImage(moving_resampled,flo_mask_file)


def generate_orienter(fixed_image, moving_image):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetMetricAsCorrelation()

    registration_method.SetOptimizerAsExhaustive(numberOfSteps=[4, 4, 4, 0, 0, 0],stepLength=np.pi / 4)
    registration_method.SetOptimizerScales([1, 1, 1, 1, 1, 1])
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform)
    return registration_method


def get_transformation_matrix(transform):
    """
    Extracts the 4x4 transformation matrix from a sitk.Transform object.

    Args:
        transform (sitk.Transform): A SimpleITK transform (e.g., Euler3DTransform).

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    # Get the 3x3 rotation matrix and translation vector
    rotation_matrix = np.array(transform.GetMatrix()).reshape(3,3)
    translation = np.array(transform.GetTranslation())

    # Construct the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix


def align_image_centers(ref_image, mov_image, output_path):
    """
    Align the centers of two SimpleITK images by adjusting the origin of the moving image.

    Args:
        ref_image (sitk.Image): The reference image.
        mov_image (sitk.Image): The moving image to be aligned.
        output_path (str): File path to save the aligned moving image.
    """
    # Compute the centers of the images in physical space
    ref_center = ref_image.TransformContinuousIndexToPhysicalPoint(
        [size / 2.0 for size in ref_image.GetSize()]
    )
    mov_center = mov_image.TransformContinuousIndexToPhysicalPoint(
        [size / 2.0 for size in mov_image.GetSize()]
    )

    # Compute the translation vector
    translation = [r - m for r, m in zip(ref_center,mov_center)]

    # Adjust the origin of the moving image
    new_origin = [o + t for o, t in zip(mov_image.GetOrigin(),translation)]
    mov_image.SetOrigin(new_origin)

    # Save the adjusted moving image
    sitk.WriteImage(mov_image,output_path)
    print(f"Aligned moving image saved to: {output_path}")


def get_centered_rotation_matrix(image, angle_degrees, rotation_plane="XZ"):
    """
    Compute a 3D transformation matrix for rotation around the image center in the specified plane.
    Args:
        image (sitk.Image): The input image.
        angle_degrees (float): The rotation angle in degrees.
        rotation_plane (str): The rotation plane ("XZ", "YZ", or "XY").
    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    # Step 1: Compute the center of the image in physical space
    size = np.array(image.GetSize())
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())

    center = origin + spacing * (size / 2.0)

    # Step 2: Create the rotation matrix
    angle_radians = np.deg2rad(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    rotation_matrix = np.eye(4)
    if rotation_plane == "XZ":
        rotation_matrix[:3, :3] = [
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ]
    elif rotation_plane == "YZ":
        rotation_matrix[:3, :3] = [
            [1, 0, 0],
            [0, cos_theta, -sin_theta],
            [0, sin_theta, cos_theta]
        ]
    elif rotation_plane == "XY":
        rotation_matrix[:3, :3] = [
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ]
    else:
        raise ValueError(f"Unsupported rotation plane: {rotation_plane}")

    # Step 3: Translate center to origin
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -center

    # Step 4: Translate back from origin
    translation_back = np.eye(4)
    translation_back[:3, 3] = center

    # Step 5: Combine transformations
    centered_rotation = np.dot(translation_back,np.dot(rotation_matrix,translation_to_origin))

    return centered_rotation


def apply_matrix_to_im(im_file, matrix):
    im = nib.load(im_file)
    im_data = im.get_fdata()
    res_aff = np.dot(matrix,im.affine)
    image_save = nib.Nifti1Image(im_data,res_aff)
    nib.save(image_save,im_file)


def rotate_image_xz(input_image, angle_degrees, center=None):
    """
    Rotate the input image by a specified angle (in degrees) in the XZ plane.
    """
    angle_radians = np.deg2rad(angle_degrees)  # Convert to radians

    # Define the rotation matrix for the XZ plane
    rotation_matrix = np.eye(3)
    rotation_matrix[0, 0] = np.cos(angle_radians)
    rotation_matrix[0, 2] = np.sin(angle_radians)
    rotation_matrix[2, 0] = -np.sin(angle_radians)
    rotation_matrix[2, 2] = np.cos(angle_radians)

    # Create the SimpleITK AffineTransform
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(rotation_matrix.flatten())

    # Set the center of rotation
    if center is None:
        center = input_image.TransformContinuousIndexToPhysicalPoint(
            [size / 2 for size in input_image.GetSize()]
        )
    transform.SetCenter(center)

    # Apply the rotation
    resampled_image = sitk.Resample(
        input_image,
        input_image,
        transform,
        sitk.sitkLinear,# Adjust interpolation method if needed
        0,# Default pixel value for outside the image
        input_image.GetPixelID()
    )

    return resampled_image, transform


def rotate_image(input_image, angle_degrees, axis="xz", center=None):
    """
    Rotate the input image by a specified angle (in degrees) along the given axis.

    Args:
        input_image (sitk.Image): The input 3D image to rotate.
        angle_degrees (float): The rotation angle in degrees.
        axis (str): The axis of rotation ("xz", "yz", "xy").
        center (tuple or None): The center of rotation. If None, uses the image center.
    """
    angle_radians = np.deg2rad(angle_degrees)  # Convert angle to radians

    # Define rotation matrices for different axes
    if axis.lower() == "xz":
        rotation_matrix = [
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)],
        ]
    elif axis.lower() == "yz":
        rotation_matrix = [
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)],
        ]
    elif axis.lower() == "xy":
        rotation_matrix = [
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1],
        ]
    else:
        raise ValueError("Invalid axis. Choose from 'xz', 'yz', or 'xy'.")

    # Flatten the rotation matrix for SimpleITK
    rotation_matrix = np.array(rotation_matrix).flatten()

    # Create the SimpleITK AffineTransform
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(rotation_matrix)

    # Set the center of rotation
    if center is None:
        center = input_image.TransformContinuousIndexToPhysicalPoint(
            [size / 2 for size in input_image.GetSize()]
        )
    transform.SetCenter(center)

    # Apply the rotation
    resampled_image = sitk.Resample(
        input_image,
        input_image,
        transform,
        sitk.sitkNearestNeighbor,# Adjust interpolation method if needed
        0,# Default pixel value for outside the image
        input_image.GetPixelID()
    )

    return resampled_image, transform


def cycle_reg(ref, fl, flm, fl_res, fl_res_msk, tp_file_out, angle, flip_axis):
    flip_image(fl,fl_res,flip_axis,output_format=sitk.sitkFloat32)
    flip_image(flm,fl_res_msk,flip_axis,output_format=sitk.sitkUInt8)

    fl_im = sitk.ReadImage(fl_res)
    fl_msk = sitk.ReadImage(fl_res_msk)

    fl_im, transf = rotate_image(fl_im,angle,axis='xy',center=None)
    sitk.WriteImage(fl_im,fl_res)
    fl_msk = sitk.Resample(fl_msk,fl_msk,transf,sitk.sitkNearestNeighbor,0.0,fl_msk.GetPixelID())
    sitk.WriteImage(fl_msk,fl_res_msk)

    is_full = coalign_roi_based(ref,fl_res,fl_res_msk,fl_res,fl_res_msk)

    if not is_full:
        print("Not full!!! ",angle,flip_axis)

    metric = perform_linear_coregistration(fl_res,ref,ref_landmarks,ref_mask,fl_res_msk,(
    fl_res, tp_file_out),inaff=None,use_landmarks=True)

    return metric


def register_linear_full(floating_image: Union[str, Path],
                         reference_image: Union[str, Path],
                         float_mask: Union[str, Path],
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

    metrics_outputs = {}

    process_folder = Path(floating_image).parent
    res_mask_path = process_folder / "coregistered_mask.nii.gz"
    if LINEAR_SIMPLE:

        perform_linear_coregistration(floating_image,reference_image,ref_landmarks,ref_mask,float_mask,outputs,inaff,use_landmarks=False)
        return

    if coalign_roi_based(
        reference_image,floating_image,
        ref_rois=config['side_ref_xml'],model_path=config['side_model'],mask_path=float_mask,
        res_path=outputs[0],
        res_mask_path=res_mask_path,out_im_name='axis_0.png',axis=0) is not None:

        metric_0 = perform_linear_coregistration(
            outputs[0],reference_image,ref_landmarks,ref_mask,res_mask_path,outputs,inaff,use_landmarks=False)
    else:
        metric_0 = perform_linear_coregistration(
            floating_image,reference_image,ref_landmarks,ref_mask,float_mask,outputs,inaff,use_landmarks=False)
    #metrics_outputs.append(metric_0)
    print("Metric 0",metric_0)


# flip_params = [ (False,False,False)]
# rot_params = [-15,0,15]
#
# out_mask =str( process_folder / "coregistered_mask.nii.gz")
#
# if metric_0< 0.5:
#
#     for i in range(len(flip_params)):
#         for j in range(len(rot_params)):
#             metric = cycle_reg(reference_image,floating_image,float_mask,outputs[0],
#                 out_mask,outputs[1],angle=rot_params[j],flip_axis=flip_params[i])
#             metrics_outputs[(i,j)] = metric
#
#             print("Metric",i,j,metric)
#
#
#     metric_0 = max(metrics_outputs.values())
#     print("Metric 0",metric_0)
#
#     # get the best transformation parameters
#
#     best_params = max(metrics_outputs,key=metrics_outputs.get)
#     print("Best params",best_params)
#
#     cycle_reg(reference_image,floating_image,float_mask,outputs[0],
#                 out_mask,outputs[1],angle=rot_params[best_params[1]],flip_axis=flip_params[best_params[0]])


def create_oriented_bounding_box(mesh: vtk.vtkPolyData):
    """
    Compute an oriented bounding box (OBB) for the input mesh.

    Args:
        mesh: vtkPolyData object representing the mesh.

    Returns:
        obb_transform: vtkTransform representing the OBB transformation.
        corner: Coordinates of the OBB corner.
        size: Dimensions of the OBB along its principal axes.
        lengths: Lengths of the OBB axes.
    """
    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(mesh)
    obb_tree.BuildLocator()

    corner = [0.0, 0.0, 0.0]
    max_point = [0.0, 0.0, 0.0]
    mid_point = [0.0, 0.0, 0.0]
    min_point = [0.0, 0.0, 0.0]
    size = [0.0, 0.0, 0.0]

    obb_tree.ComputeOBB(mesh,corner,max_point,mid_point,min_point,size)

    # Compute the lengths of the OBB axes
    lengths = [
        vtk.vtkMath.Distance2BetweenPoints(corner,max_point) ** 0.5,
        vtk.vtkMath.Distance2BetweenPoints(corner,mid_point) ** 0.5,
        vtk.vtkMath.Distance2BetweenPoints(corner,min_point) ** 0.5,
    ]

    # Compute the OBB transformation matrix
    axes = [
        [max_point[i] - corner[i] for i in range(3)],
        [mid_point[i] - corner[i] for i in range(3)],
        [min_point[i] - corner[i] for i in range(3)],
    ]

    for i in range(3):
        vtk.vtkMath.Normalize(axes[i])

    obb_matrix = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            obb_matrix.SetElement(i,j,axes[j][i])
        obb_matrix.SetElement(i,3,corner[i])

    obb_matrix.SetElement(3,0,0)
    obb_matrix.SetElement(3,1,0)
    obb_matrix.SetElement(3,2,0)
    obb_matrix.SetElement(3,3,1)

    obb_transform = vtk.vtkTransform()
    obb_transform.SetMatrix(obb_matrix)

    return obb_transform, corner, size, lengths


def crop_image_with_obb(input_image_path: Path, output_image_path: Path, obb_transform: vtk.vtkTransform, lengths: list):
    """
    Crop a vtkImageData using an Oriented Bounding Box (OBB).

    Args:
        input_image_path: Path to the input image.
        output_image_path: Path to save the cropped image.
        obb_transform: The vtkTransform of the OBB.
        lengths: The dimensions of the OBB along its principal axes.

    Returns:
        None. The cropped image is saved to the output path.
    """
    # Read the input image
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(input_image_path))
    reader.Update()

    image = reader.GetOutput()

    # Create an oriented implicit box function
    box = vtk.vtkBox()
    bounds = [-lengths[0] / 2, lengths[0] / 2,
              -lengths[1] / 2, lengths[1] / 2,
              -lengths[2] / 2, lengths[2] / 2]
    box.SetBounds(bounds)

    # Apply the OBB transform
    implicit_transform = vtk.vtkTransform()
    implicit_transform.DeepCopy(obb_transform)
    box.SetTransform(implicit_transform)

    # Use vtkImageStencil to crop the image
    stencil = vtk.vtkImplicitFunctionToImageStencil()
    stencil.SetInput(box)
    stencil.SetOutputOrigin(image.GetOrigin())
    stencil.SetOutputSpacing(image.GetSpacing())
    stencil.SetOutputWholeExtent(image.GetExtent())
    stencil.Update()

    image_stencil = vtk.vtkImageStencil()
    image_stencil.SetInputData(image)
    image_stencil.SetStencilConnection(stencil.GetOutputPort())
    image_stencil.ReverseStencilOff()
    image_stencil.SetBackgroundValue(0)
    image_stencil.Update()

    # Write the cropped image
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(str(output_image_path))
    writer.SetInputData(image_stencil.GetOutput())
    writer.Write()


def save_mesh(mesh: vtk.vtkPolyData, output_path: Path):
    """
    Save a vtkPolyData mesh to a file.

    Args:
        mesh: The vtkPolyData mesh.
        output_path: Path to save the mesh file.

    Returns:
        None. The mesh is saved to the output path.
    """
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(mesh)
    writer.Write()


def create_bounding_box_mask(image_path: Path, output_path: Path, obb_transform: vtk.vtkTransform, lengths: list):
    """
    Create a binary mask of the oriented bounding box and save it as a NIfTI file.

    Args:
        image_path: Path to the input image.
        output_path: Path to save the binary mask.
        obb_transform: The vtkTransform of the OBB.
        lengths: Dimensions of the OBB along its principal axes.
    """
    # Read the input image
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(image_path))
    reader.Update()

    image = reader.GetOutput()

    # Create an oriented implicit box function
    box = vtk.vtkBox()
    bounds = [-lengths[0] / 2, lengths[0] / 2,
              -lengths[1] / 2, lengths[1] / 2,
              -lengths[2] / 2, lengths[2] / 2]
    box.SetBounds(bounds)

    # Apply the OBB transform
    implicit_transform = vtk.vtkTransform()
    implicit_transform.DeepCopy(obb_transform)
    box.SetTransform(implicit_transform)

    # Generate a binary mask using vtkImageStencil
    stencil = vtk.vtkImplicitFunctionToImageStencil()
    stencil.SetInput(box)
    stencil.SetOutputOrigin(image.GetOrigin())
    stencil.SetOutputSpacing(image.GetSpacing())
    stencil.SetOutputWholeExtent(image.GetExtent())
    stencil.Update()

    image_stencil = vtk.vtkImageStencil()
    image_stencil.SetInputData(image)
    image_stencil.SetStencilConnection(stencil.GetOutputPort())
    image_stencil.ReverseStencilOff()
    image_stencil.SetBackgroundValue(0)
    image_stencil.Update()

    # Write the binary mask as a NIfTI file
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(image_stencil.GetOutput())
    writer.Write()


def simplify_mesh(mesh, reduction_factor=0.5):
    """
    Simplify a mesh by reducing the number of points and triangles.

    Args:
        mesh: vtkPolyData input mesh.
        reduction_factor: Fraction of the original points to keep (0.0 to 1.0).

    Returns:
        vtkPolyData: Simplified mesh.
    """
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(mesh)
    decimate.SetTargetReduction(1.0 - reduction_factor)  # Reduction is the fraction to remove
    decimate.PreserveTopologyOn()
    decimate.Update()
    return decimate.GetOutput()


def create_wrapping_mesh(mesh, scale_factor=1.1):
    """
    Create a tightly wrapping mesh from the input mesh using vtkDelaunay3D and vtkGeometryFilter.

    Args:
        mesh: vtkPolyData input mesh.
        scale_factor: Scale the resulting wrapping mesh by this factor.

    Returns:
        vtkPolyData: The wrapping mesh.
    """
    if mesh.GetNumberOfPoints() == 0 or mesh.GetNumberOfCells() == 0:
        raise ValueError("Input mesh is empty or invalid. Cannot create wrapping mesh.")

    # Perform Delaunay triangulation to wrap the mesh
    delaunay = vtk.vtkDelaunay3D()
    # Create a vtkPolyData object to hold the points
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(mesh.GetPoints())

    delaunay.SetInputData(mesh)
    delaunay.SetAlpha(0.1)  # Use alpha shape to control tightness of the wrapping
    delaunay.Update()
    print("Delaunay triangulation done")
    # Extract the surface geometry from the 3D Delaunay triangulation
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputConnection(delaunay.GetOutputPort())
    geometry_filter.Update()

    wrapped_mesh = geometry_filter.GetOutput()

    if wrapped_mesh.GetNumberOfPoints() == 0:
        raise ValueError("Wrapped mesh is empty. Delaunay triangulation may have failed.")

    # Apply scaling to expand the wrapping mesh
    transform = vtk.vtkTransform()
    transform.Scale(scale_factor,scale_factor,scale_factor)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(wrapped_mesh)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    scaled_wrapped_mesh = transform_filter.GetOutput()

    if scaled_wrapped_mesh.GetNumberOfPoints() == 0:
        raise ValueError("Scaled wrapped mesh is empty. Transformation may have failed.")

    return scaled_wrapped_mesh


def create_bounding_box_mask_wrapped(
        image_path: Path,
        output_path: Path,
        wrapping_mesh: vtk.vtkPolyData
):
    """
    Create a binary mask of the wrapping mesh and save it as a NIfTI file.

    Args:
        image_path: Path to the input image.
        output_path: Path to save the binary mask.
        wrapping_mesh: vtkPolyData of the wrapping mesh.
    """
    # Read the input image
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(image_path))
    reader.Update()

    image = reader.GetOutput()

    # Convert the mesh to an image stencil
    polydata_to_image_stencil = vtk.vtkPolyDataToImageStencil()
    polydata_to_image_stencil.SetInputData(wrapping_mesh)
    polydata_to_image_stencil.SetOutputSpacing(image.GetSpacing())
    polydata_to_image_stencil.SetOutputOrigin(image.GetOrigin())
    polydata_to_image_stencil.SetOutputWholeExtent(image.GetExtent())
    polydata_to_image_stencil.Update()

    # Apply the stencil to the image
    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image)
    stencil.SetStencilData(polydata_to_image_stencil.GetOutput())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)  # Set voxels outside the mesh to 0
    stencil.Update()

    # Write the mask as a NIfTI file
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(stencil.GetOutput())
    writer.Write()


def create_wrapping_mesh_with_alpha_shape(mesh, alpha=1.0):
    """
    Create a wrapping mesh using vtkAlphaShape.

    Args:
        mesh: vtkPolyData input mesh.
        alpha: Alpha value for controlling the tightness of the wrapping. Smaller values result in tighter wrapping.

    Returns:
        vtkPolyData: The wrapping mesh.
    """
    if mesh.GetNumberOfPoints() == 0:
        raise ValueError("Input mesh has no points. Cannot create wrapping mesh.")

    # Extract points from the input mesh
    points = vtk.vtkPoints()
    points.DeepCopy(mesh.GetPoints())

    # Create a new vtkPolyData to hold the points
    point_polydata = vtk.vtkPolyData()
    point_polydata.SetPoints(points)

    # Apply vtkAlphaShape for wrapping
    alpha_shape = vtk.vtkAlphaShape()
    alpha_shape.SetInputData(point_polydata)
    alpha_shape.SetAlpha(alpha)  # Adjust this parameter to control wrapping tightness
    alpha_shape.Update()

    # Check if the output is valid
    wrapped_mesh = alpha_shape.GetOutput()
    if wrapped_mesh.GetNumberOfPoints() == 0 or wrapped_mesh.GetNumberOfCells() == 0:
        raise ValueError("Alpha shape failed to produce a valid wrapping mesh. Try adjusting the alpha parameter.")

    return wrapped_mesh


def create_bounding_box_mask_wrapped(image_path, output_path, wrapping_mesh):
    """
    Create a binary mask using a wrapping mesh (oriented bounding box) and save it as a NIFTI image.

    Args:
        image_path (str or Path): Path to the input NIFTI image.
        output_path (str or Path): Path to save the binary mask.
        wrapping_mesh (vtkPolyData): The wrapping mesh to use for masking.
    """
    # Read the input image
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(image_path))
    reader.Update()
    image = reader.GetOutput()

    # Get image properties
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    dimensions = image.GetDimensions()

    # Create a vtkImageStencil from the wrapping mesh
    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(wrapping_mesh)
    poly_to_stencil.SetOutputSpacing(spacing)
    poly_to_stencil.SetOutputOrigin(origin)
    poly_to_stencil.SetOutputWholeExtent(image.GetExtent())
    poly_to_stencil.Update()

    # Apply the stencil to the image
    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image)
    stencil.SetStencilConnection(poly_to_stencil.GetOutputPort())
    stencil.ReverseStencilOff()  # Keep only the inside of the wrapping mesh
    stencil.SetBackgroundValue(0)  # Set outside values to 0
    stencil.Update()

    # Write the result to a NIFTI file
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(stencil.GetOutput())
    writer.Write()


def create_bounding_box_mask_ras(image_path, output_path, wrapping_mesh):
    """
    Create a binary mask from a wrapping mesh (in RAS space) and save it as a NIFTI file.

    Args:
        image_path (str or Path): Path to the input NIFTI image.
        output_path (str or Path): Path to save the binary mask.
        wrapping_mesh (vtkPolyData): The wrapping mesh in RAS space.
    """
    import vtk
    import numpy as np

    # Read the input NIfTI image
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(image_path))
    reader.Update()
    image = reader.GetOutput()

    # Image properties
    origin = image.GetOrigin()  # Physical origin
    spacing = image.GetSpacing()  # Voxel size
    dimensions = image.GetDimensions()  # Voxel grid size

    # Debug: Print image properties
    print(f"Image Origin (RAS): {origin}, Spacing: {spacing}, Dimensions: {dimensions}")

    # Create a stencil for the mesh
    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(wrapping_mesh)
    poly_to_stencil.SetOutputSpacing(spacing)  # Match the image spacing
    poly_to_stencil.SetOutputOrigin(origin)  # Match the image origin
    poly_to_stencil.SetOutputWholeExtent(image.GetExtent())  # Match the image extent
    poly_to_stencil.Update()

    # Apply the stencil to the image
    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image)
    stencil.SetStencilConnection(poly_to_stencil.GetOutputPort())
    stencil.ReverseStencilOff()  # Keep only the inside of the mesh
    stencil.SetBackgroundValue(0)  # Background voxels set to 0
    stencil.Update()

    # Write the result to a NIfTI file
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(stencil.GetOutput())
    writer.Write()

    print(f"Binary mask saved to {output_path}")


def convert_mesh_to_lps(mesh):
    transform = vtk.vtkTransform()
    transform.Scale(-1,-1,1)  # RAS to LPS conversion
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(mesh)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    return transform_filter.GetOutput()


def create_bounding_box_mask_ras(image_path, output_path, wrapping_mesh):
    """
    Create a binary mask from a wrapping mesh (in RAS space) and save it as a NIFTI file.
    """
    import vtk
    import numpy as np

    # Read the input NIfTI image
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(image_path))
    reader.Update()
    image = reader.GetOutput()

    # Image properties
    origin = image.GetOrigin()  # Physical origin
    spacing = image.GetSpacing()  # Voxel size
    dimensions = image.GetDimensions()  # Voxel grid size

    # Debug: Print image properties
    print(f"Image Origin (RAS): {origin}")
    print(f"Image Spacing: {spacing}")
    print(f"Image Dimensions: {dimensions}")

    print("bounds",
        origin[0] + dimensions[0] * spacing[0],
        origin[0],origin[1] + dimensions[1] * spacing[1],origin[1],origin[2] + dimensions[2] * spacing[2],origin[2])

    # Mesh properties
    bounds = wrapping_mesh.GetBounds()
    print("Mesh Bounds (RAS):",bounds)

    # Check for overlap
    overlap = all([
        bounds[0] <= origin[0] + dimensions[0] * spacing[0] and bounds[1] >= origin[0],
        bounds[2] <= origin[1] + dimensions[1] * spacing[1] and bounds[3] >= origin[1],
        bounds[4] <= origin[2] + dimensions[2] * spacing[2] and bounds[5] >= origin[2],
    ])

    if not overlap:
        print("ERROR: The mesh does not overlap with the image. Mask will be empty.")
        return

    # Create a stencil for the mesh
    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(wrapping_mesh)
    poly_to_stencil.SetOutputSpacing(spacing)  # Match the image spacing
    poly_to_stencil.SetOutputOrigin(origin)  # Match the image origin
    poly_to_stencil.SetOutputWholeExtent(image.GetExtent())  # Match the image extent
    poly_to_stencil.Update()

    # Apply the stencil to the image
    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image)
    stencil.SetStencilConnection(poly_to_stencil.GetOutputPort())
    stencil.ReverseStencilOff()  # Keep only the inside of the mesh
    stencil.SetBackgroundValue(0)  # Background voxels set to 0
    stencil.Update()

    # Write the result to a NIfTI file
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(stencil.GetOutput())
    writer.Write()

    print(f"Binary mask saved to {output_path}")


def simplify_mesh_quadric(mesh, reduction_factor=0.5):
    """
    Simplify a mesh using vtkQuadricDecimation.

    Args:
        mesh: vtkPolyData input mesh.
        reduction_factor: Fraction of the original points to retain (0.0 to 1.0).

    Returns:
        vtkPolyData: Simplified mesh.
    """
    target_reduction = 1.0 - reduction_factor  # Fraction to remove
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(mesh)
    decimate.SetTargetReduction(target_reduction)
    decimate.Update()
    return decimate.GetOutput()


def create_wrapping_mesh_with_delaunay(mesh, alpha=0.0, scale_factor=1.1):
    """
    Create a wrapping mesh using vtkDelaunay3D.

    Args:
        mesh: vtkPolyData input mesh.
        alpha: Alpha value for controlling the Delaunay triangulation (set to 0 for tight wrapping).
        scale_factor: Scale the wrapping mesh for optional padding.

    Returns:
        vtkPolyData: The wrapping mesh.
    """
    if mesh.GetNumberOfPoints() == 0 or mesh.GetNumberOfCells() == 0:
        raise ValueError("Input mesh is empty or invalid. Cannot create wrapping mesh.")

    # Perform Delaunay triangulation
    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(mesh)
    delaunay.SetAlpha(alpha)  # Use alpha value for tightness control
    delaunay.AlphaTetsOn()
    delaunay.Update()

    # Extract surface geometry from Delaunay output
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputConnection(delaunay.GetOutputPort())
    geometry_filter.Update()

    wrapped_mesh = geometry_filter.GetOutput()

    if wrapped_mesh.GetNumberOfPoints() == 0 or wrapped_mesh.GetNumberOfCells() == 0:
        raise ValueError("Delaunay triangulation failed to produce a valid wrapping mesh. Adjust the alpha parameter.")

    # Compute the mesh's center of mass
    center_filter = vtk.vtkCenterOfMass()
    center_filter.SetInputData(wrapped_mesh)
    center_filter.Update()
    mesh_center = center_filter.GetCenter()

    # Translate the mesh to the origin
    move_to_origin_transform = vtk.vtkTransform()
    move_to_origin_transform.Translate(-mesh_center[0],-mesh_center[1],-mesh_center[2])

    move_to_origin_filter = vtk.vtkTransformPolyDataFilter()
    move_to_origin_filter.SetInputData(wrapped_mesh)
    move_to_origin_filter.SetTransform(move_to_origin_transform)
    move_to_origin_filter.Update()

    mesh_at_origin = move_to_origin_filter.GetOutput()

    # Scale the mesh at the origin
    scale_transform = vtk.vtkTransform()
    scale_transform.Scale(scale_factor,scale_factor,scale_factor)

    scale_filter = vtk.vtkTransformPolyDataFilter()
    scale_filter.SetInputData(mesh_at_origin)
    scale_filter.SetTransform(scale_transform)
    scale_filter.Update()

    scaled_mesh_at_origin = scale_filter.GetOutput()

    # Translate the mesh back to its original position
    move_back_transform = vtk.vtkTransform()
    move_back_transform.Translate(mesh_center[0],mesh_center[1],mesh_center[2])

    move_back_filter = vtk.vtkTransformPolyDataFilter()
    move_back_filter.SetInputData(scaled_mesh_at_origin)
    move_back_filter.SetTransform(move_back_transform)
    move_back_filter.Update()

    final_wrapped_mesh = move_back_filter.GetOutput()

    return final_wrapped_mesh


# Rule implementation
rule otsu_threshold:
    input:
        (OUT_FOLD / "{subject}/make_nifti_image.flag")
    output:
        flag=(OUT_FOLD / "{subject}/1_otsu.flag")
    threads: 1

    run:
        inp_file = Path(output.flag).parent / "ct_image.nii.gz"
        inp_file2 = Path(output.flag).parent / "ct_image_clean.nii.gz"
        out_file = Path(output.flag).parent / "bone_coregistration_mask.nii.gz"
        out_file.unlink(missing_ok=True)

        mesh_output = Path(output.flag).parent / "obb_mesh.obj"
        bb_mask_output = Path(output.flag).parent / "obb_mask.nii.gz"

        if inp_file.exists():
            # Apply otsu threshold

            if OTSU_REMOVE_OUTER:
                inp_img = sitk.ReadImage(str(inp_file))
                inp_img.SetOrigin([0, 0, 0])
                inp_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
                sitk.WriteImage(inp_img,str(inp_file2),useCompression=True)
                otsu_treshold(inp_file2,out_file)

                # Generate marching cubes mesh
                mesh, _ = marching_cubes(out_file)
                #save_mesh(mesh,mesh_output)

                # Create a wrapping mesh
                wrapped_mesh = simplify_mesh_quadric(mesh,reduction_factor=0.05)
                wrapped_mesh = create_wrapping_mesh_with_delaunay(wrapped_mesh,alpha=0.0)
                wrapped_mesh = convert_mesh_to_lps(wrapped_mesh)

                create_bounding_box_mask_ras(inp_file2,inp_file2,wrapped_mesh)
                otsu_treshold(inp_file2,out_file,origin_copy=inp_file)
            else:
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
Parameters: {' '.join(map(str,rotation_flat))} {' '.join(map(str,translation_flat))}
FixedParameters: 0 0 0
"""

    # Write the content to the file
    with open(filename,'w') as file:
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
            params_str = line.replace("(TransformParameters","").replace(")","").strip()
            transform_params = list(map(float,params_str.split()))
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


def reorient_normal_image(image_orig: sitk.Image, label_orig: sitk.Image):
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(label_orig)

    image_result = sitk.Image(image_orig)
    direction = label_shape_filter.GetOrientedBoundingBoxDirection(1)
    direction = np.reshape(direction,(3, 3))

    direction[:, 0] = -1 * direction[:, 0]
    print(direction.reshape(-1))
    direction = np.array(direction,dtype=float)
    print("Direction",direction)

    image_result.SetDirection(direction.flatten().tolist())
    # if get_frontal_voxels(label_orig) > get_frontal_voxels(label_orig,False):
    #     print(True)
    # return
    #image_result.SetDirection(direction)
    image_result.SetOrigin(label_shape_filter.GetCentroid(1))

    return image_result


rule linear_coregistration:
    input: OUT_FOLD / "{subject}/1_otsu.flag"
    output: OUT_FOLD / "{subject}/2_lin_cor.flag"
    params:
        in_ct=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/ct_image.nii.gz",
        in_landmarks=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/landmarks.pts",
        in_mask=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/bone_coregistration_mask.nii.gz",
        out_head=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/coregistered_head.nii.gz",
        out_mat=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/rigid_head.txt",
        out_affine=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/affine_head.txt"

    threads: 2
    run:
        process_folder = Path(output[0]).parent
        Path(str(params.out_head)).unlink(missing_ok=True)
        Path(str(params.out_mat)).unlink(missing_ok=True)
        Path(str(params.out_affine)).unlink(missing_ok=True)

        register_linear_full(str(params.in_ct),str(ref_im),
            str(params.in_mask),ref_mask=str(ref_mask),

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


def non_rigid_registration(flo_im: str, ref_im: str, out_transform: str, out_image: str, param_file: str):
    # Reading
    fold = Path(out_image).parent
    cmd = f"elastix -f '{ref_im}' -m '{flo_im}' -p '{param_file}' -out '{str(fold)}'"
    shell(cmd)
    shell(f"mv -f '{str(fold / 'result.0.nii.gz')}' '{out_image}'")
    shell(f"mv -f '{str(fold / 'TransformParameters.0.txt')}' '{out_transform}'")


rule non_rigid_registration_rule:
    input: (OUT_FOLD / "{subject}/2_lin_cor.flag")
    output: (OUT_FOLD / "{subject}/3_nonrig.flag")
    params:
        out_nlim=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/nonlin_image.nii.gz",
        out_txt=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/nl_transformation_params.txt",
        inp_coregim=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/coregistered_head.nii.gz"
    threads: 2
    run:
        Path(output[0]).touch(exist_ok=True)
        # remove previous files
        Path(str(params.out_nlim)).unlink(missing_ok=True)
        Path(str(params.out_txt)).unlink(missing_ok=True)

        if not params.inp_coregim.exists():
            return
        # script coregisters reference image (choosen as default to each linearly registered head for reverse mesh generation
        or_ref_im = REF_IMAGE / 'oriented_image.nii.gz'
        non_rigid_registration(flo_im=str(or_ref_im),ref_im=str(params.inp_coregim),
            out_transform=str(params.out_txt),out_image=str(params.out_nlim),param_file=str(PARAMETER_FILE_NONLIN))


rule inverse_transform_mesh_rule:
    input: (OUT_FOLD / "{subject}/3_nonrig.flag")
    output: (OUT_FOLD / "{subject}/4_inv_mesh.flag")
    params:
        in_ct=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/nl_transformation_params.txt",
        out_head=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/head_affine_registered.obj",

    threads: 2
    run:
        # remove previous files
        Path(str(params.out_head)).unlink(missing_ok=True)
        dir1name = str(Path(input[0]).parent)
        Path(output[0]).touch(exist_ok=True)
        #generate deformation field
        if not Path(params.in_ct).exists():
            return
        cmd = f"transformix -tp '{params.in_ct}' -out '{dir1name}' -def all"
        shell(cmd)
        or_ref_mesh = REF_IMAGE / "oriented_mesh.obj"
        print(or_ref_mesh)

        def_field_file = dir1name + os.sep + "deformationField.nii.gz"
        ###deform original mesh

        cmd = (f"xvfb-run -a  {SLICER_CORE} --python-script {MESH_TRANSFORM_SCRIPT} '{def_field_file}' '{or_ref_mesh}' "
               f"'{params.out_head}'  --testing --no-splash --no-main-window")
        print(cmd)
        shell(cmd)
        shell("rm -rf '" + def_field_file + "'")


### rule for computing of mahalanobis pic's results

def read_coh_file():
    coh_file_l = open(coh_file,'rb')
    COHORTS = pickle.load(coh_file_l)

    coh_file_l.close()
    return COHORTS
    print("FILE READ")


def sutures_open(image_file, mesh_file):
    model = SuturesNetWrapper(config['sutures_model'])
    return model.predict(image_file,mesh_file)


rule general_procrustes:
    input: (OUT_FOLD / "{subject}/6_analyse_scalling.flag")
    output: (OUT_FOLD / "{subject}/6_procr_result.flag")
    params:
        in_ct=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/mahalanobis.mat",
        in_image=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/coregistered_head.nii.gz",
        in_matr=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/result_head.mat",
        head=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/head_affine_registered.obj",
        out_file=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/abnormality_automated.csv",
    run:
        Path(output[0]).touch(exist_ok=True)
        # remove previous files
        Path(str(params.out_file)).unlink(missing_ok=True)

        if Path(params.head).exists():
            read_orig_mesh = read_vtk_obj(str(params.head))
            ref_mesh_pts = get_mesh_vertices(read_orig_mesh) - np.array(center_of_mesh(read_orig_mesh))

            # read normal points
            cohorts = read_coh_file()
            subject_gender = Path(output[0]).parent.name.split("_")[1]
            ref_mesh, _, _ = gpa(cohorts[subject_gender])

            _, distance, _ = procreustes_alignment(ref_mesh,ref_mesh_pts)

            #load 99 percentiles and compute percentage
            face_perc_file = MALE_FACE_PERC if subject_gender == 'M' else FEMALE_FACE_PERC

            face_perc_data = scio.loadmat(face_perc_file)
            perc99 = face_perc_data['perc99']
            energies = scio.loadmat(str(params.in_ct))['energies']
            ratio = (energies > perc99).sum() / perc99.shape[1]
            is_suture = sutures_open(params.in_image,params.head)

            ##
            mat = np.loadtxt(params.in_matr)
            u, s, v = np.linalg.svd(mat)

            array = {"proc_dist": distance,
                     "abn_distance": "normal" if distance < 980 else "abnormal", "sutures_open": is_suture,
                     "area99perc": ratio, "abn_perc": "normal" if ratio < 0.1 else "abnormal", "s_x": s[0], "s_y": s[1],
                     "s_z": s[2]}
            df = pd.Series(array)
            df.to_csv(params.out_file)


rule combine_results:
    input: expand(OUT_FOLD / "{subject}/7_gifs.flag",subject=SRC_SUBJECTS)
    output: (PROCESS_DIR / "abnormality_automated_combined.csv")
    params:
        out_file=OUT_FOLD / "abnormality_automated_combined.csv"
    run:
        columns = ["proc_dist", "abn_distance",
                   "area99perc", "abn_perc", 'sutures_open', "s_x", "s_y", "s_z"]
        index = []
        res = []
        for fold in input:
            p_fold = Path(fold).parent
            index.append(p_fold.name)
            if (p_fold / "abnormality_automated.csv").exists():
                subject = pd.read_csv((p_fold / "abnormality_automated.csv"),header=0,index_col='Unnamed: 0')
                subject = subject.transpose().iloc[0]

                res.append(dict(subject))
            else:
                res.append({x: None for x in columns})
        print(res)
        pd.DataFrame(res,index=index).to_csv(output[0])


def read_mesh_and_points(obj_filename):
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


def create_mah_gif(input_mat_file, input_obj_file, output_gif_file, title):
    # Load the energies
    cmp_str = 'jet'
    energies = scio.loadmat(input_mat_file)['energies']
    energies[np.where(energies > 100)] = 100
    energies = energies[0]
    print(energies[0])
    # Load the mesh
    pts, faces, mesh = read_mesh_and_points(input_obj_file)

    # Create a figure
    fig = plt.figure(figsize=(8, 6),dpi=400)
    ax = fig.add_subplot(111,projection='3d')

    # Set the colormap (adjust as desired)
    norm = matplotlib.colors.Normalize(0,100)
    m = cm.ScalarMappable(cmap=plt.cm.get_cmap(cmp_str),norm=norm)
    m.set_array([])
    print((norm(energies)))

    fig.colorbar(m,ax=ax)

    cmap = plt.get_cmap(cmp_str)
    norm = plt.Normalize(0,100)
    cmap = plt.get_cmap(cmp_str)
    p3dc = ax.plot_trisurf(pts[:, 0],pts[:, 1],pts[:, 2],
        triangles=faces,linewidth=0.2  #,)
    )
    ax.set_box_aspect([np.ptp(i) for i in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    p3dc.set_fc(cmap(norm(energies)))

    #ax.set_autoscale_on(False)
    # plt.savefig(output_gif_file.split(".")[0] + ".png")
    # Create the animation function
    def animate(frame):
        ax.set_box_aspect([np.ptp(i) for i in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
        p3dc = ax.plot_trisurf(pts[:, 0],pts[:, 1],pts[:, 2],
            triangles=faces,linewidth=4)  #,)

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
    ani = animation.FuncAnimation(fig,animate,frames=num_frames)
    print("Attempting to save gif to ",output_gif_file)
    # Save the animation as a GIF
    ani.save(output_gif_file,writer='pillow',fps=1)

    plt.close(fig)
    print('GIF saved successfully.')


def create_perc_gif(input_mat_file, input_obj_file, output_fig_file, face_perc_file, title=None):
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
        if colormap_values[0, i] == 1:
            cmp.append('blue')
        elif colormap_values[0, i] == 2:
            cmp.append('green')
        elif colormap_values[0, i] == 3:
            cmp.append('yellow')
        elif colormap_values[0, i] == 4:
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
    ax = fig.add_subplot(111,projection='3d')

    # Define the colormap with RGBA values

    p3dc = ax.plot_trisurf(pts[:, 0],pts[:, 1],pts[:, 2],
        triangles=faces,linewidth=0.20)

    # Ensure the number of face colors matches the number of faces
    if len(cmp) != len(faces):
        print("Number of face colors doesn't match number of faces.",len(cmp),len(faces))
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
    ani = animation.FuncAnimation(fig,animate,frames=num_frames)

    # Save the animation as a GIF
    ani.save(output_fig_file,writer='pillow',fps=1)

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
        points = points[
            np.where(mahalanobis_distances < np.percentile(mahalanobis_distances,99))[0]]  # smart thresholding?
        emp_cov.fit(points)
        mahalanobis_distance = emp_cov.mahalanobis([mesh_pts[i]])[0]
        dists.append(mahalanobis_distance)

    return dists


rule compute_mahalanobis_heads:
    input: (OUT_FOLD / "{subject}/4_inv_mesh.flag")
    output: (OUT_FOLD / "{subject}/5_mah_mat.flag")
    params:
        in_ct=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/nl_transformation_params.txt",
        head=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/head_affine_registered.obj",
        out_file=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/mahalanobis.mat",
    threads: 1
    run:
        COHORTS = read_coh_file()
        par_name = Path(input[0]).parent.name
        Path(output[0]).touch(exist_ok=True)
        if Path(params.head).exists():
            cohort = COHORTS[par_name.split("_")[1]]  # copy males or females

            mesh_pts, faces, mesh = read_mesh_and_points(params.head)

            com = center_of_mesh(mesh)
            cohort[par_name] = mesh_pts - np.array(com)

            #test_V = np.stack(cohort.values())
            _, _, shapes = gpa(cohort,max_iterations=100,threshold=1e-11)

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
    input: OUT_FOLD / "{subject}/6_procr_result.flag"
    output: (OUT_FOLD / "{subject}/7_gifs.flag")
    params:
        in_mat=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/mahalanobis.mat",
        head=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/head_affine_registered.obj",
        out_mah_file=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/mahalanobis.gif",
        out_perc_file=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/perc.gif",
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
    input: OUT_FOLD / "{subject}/5_mah_mat.flag"
    output: (OUT_FOLD / "{subject}/6_analyse_scalling.flag")
    params:
        in_mat1=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/mahalanobis.mat",
        head=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/head_affine_registered.obj",
        in_mat=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/affine_head.txt",
        out_mat=lambda wildcards: OUT_FOLD / f"{wildcards.subject}/result_head.mat"
    threads: 1
    run:
        Path(output[0]).touch(exist_ok=True)
        if Path(params.in_mat).exists():
            or_ref_ = REF_IMAGE / "oriented_mesh.obj"
            read_orig_mesh = read_vtk_obj(str(or_ref_))
            ref_mesh_pts = get_mesh_vertices(read_orig_mesh) - np.array(center_of_mesh(read_orig_mesh))
            input_mesh = read_vtk_obj(params.head)
            input_mesh_pts = get_mesh_vertices(input_mesh) - np.array(center_of_mesh(input_mesh))

            _, _, matrix = procreustes_alignment(ref_mesh_pts,input_mesh_pts)

            or_ref_im = REF_IMAGE / "oriented_image.nii.gz"
            print("outmat",matrix)
            prev_mat = np.loadtxt(params.in_mat)
            prev_mat = prev_mat[:3, :3]

            print(prev_mat)
            out_mat = np.dot(matrix,prev_mat)

            np.savetxt(params.out_mat,out_mat)


def analyse_dirs_logs(folder: Union[str, Path]):
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
        (PROCESS_DIR / "abnormality_automated_combined.csv")
    output:
        (PROCESS_DIR / "logs_skull.txt")
    run:
        logstr = ""
        for x in SRC_SUBJECTS:
            fold = OUT_FOLD / x
            logstr += "Subject " + str(x) + ": " + str(analyse_dirs_logs(fold))
        Path(output[0]).touch(exist_ok=True)
        Path(output[0]).write_text(logstr)
