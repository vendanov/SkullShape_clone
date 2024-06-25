import os
import os
import pickle
import shlex
import subprocess
import sys
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Dict, Tuple, List

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

sys.path.append(  str(Path(workflow.basedir).parent / 'scripts'))
from gpa import *
from vtk_helper import *
PROCESS_DIR = Path(config['cohort_dir'])

##global vars
REF_IMAGE = Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/reference_image/')

PARAMETER_FILE_NONLIN = Path("/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/Parameters_Stable_NL.txt")
MESH_TRANSFORM_SCRIPT = "/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/scripts/support/convert_mesh.py"
SLICER_CORE = config['slicer_home']#"/home/varhai/Slicer-5.2.2-linux-amd64/Slicer"

MALE_FACE_PERC = config['male_perc']#r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/wt_male_f_percentiles.mat"
FEMALE_FACE_PERC = config['female_perc']#r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/wt_female_f_percentiles.mat"
coh_file  = config['cohort_pickle']#r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/wt_cohorts_pts.pkl"

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
    not_cohort = ["BMD","TMD","Morphology","csv","+","workdir","checkpoints","_Screen"]
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

def flip_image(im_file_in: str, im_file_out: str, axis:Tuple[int,int,int]):
    # Load your image (replace 'image_path' with the path to your image)
    image = sitk.ReadImage(im_file_in)
    # Flip the image along a specific axis (e.g., along the x-axis)
    # Flip the image along the x-axis
    #flipped_image = sitk.Flip(image, [True, False, False])  # [X, Y, Z] axis flipping
    # Specify the axis along which you want to mirror (0 for x-axis, 1 for y-axis, 2 for z-axis)
    axis_to_mirror = 0  # For example, mirror along the x-axis
    image_array = sitk.GetArrayFromImage(image)
    # Flip the image along the specified axis
    mirrored_image_array = np.flip(image_array, axis=axis#'(1,2)
    )

    # Create a SimpleITK image from the mirrored NumPy array
    flipped_image = sitk.GetImageFromArray(mirrored_image_array)

    # Set the origin, spacing, and direction cosine matrix if needed
    flipped_image.SetOrigin(image.GetOrigin())
    flipped_image.SetSpacing(image.GetSpacing())
    flipped_image.SetDirection(image.GetDirection())

    # Apply the flipping transformation to the image

    sitk.WriteImage(flipped_image, im_file_out)

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

    else:
        cmd = [ 'reg_aladin','-ref',
                str(reference_image), '-rmask', ref_mask,
                '-flo', floating_image,'-fmask',float_mask,
                '-speeeed','-cog','-res',str(outputs[0]),'-aff', str(outputs[1])
            , '-inaff',str(inaff)]
        cmd = "reg_aladin -ref '{refim}' -rmask '{rmask}' -flo '{flim}' -fmask '{fmask}'".format(refim=reference_image,
            rmask=ref_mask,flim=floating_image,fmask=float_mask)
        cmd += " -speeeeed -cog -res '" + str(outputs[0])
        cmd += "' -aff '" + str(outputs[1]) + "' -inaff '" + str(inaff) + "'"

    print(cmd)
    subprocess.call(shlex.split(cmd))
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


rule linear_coregistration:
    input: OUT_FOLD/"{subject}/1_otsu.flag"
    output: OUT_FOLD/"{subject}/2_lin_cor.flag"
    params:
        in_ct = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/ct_image.nii.gz",
        in_mask = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/bone_coregistration_mask.nii.gz",
        out_head = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/coregistered_head.nii.gz",
        out_mat = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/affine_head.mat"
    threads: 2
    run:

        if not(Path(params.in_ct).exists() and Path(params.in_mask).exists()):
            Path(output[0]).touch(exist_ok=True)
            print("FILES DOES NOT EXIST!!!! : {x1} {x2}".format(x1 = Path(params.in_ct).exists(), x2 =Path(params.in_mask).exists() ))
            print(params.in_ct)
            print(params.in_mask)
            return
        proces_folder = Path(input[0]).parent
        out_file = Path(output[0])

        ref_mask_1 = REF_IMAGE/"oriented_mask.nii.gz"
        or_ref_im = REF_IMAGE / 'oriented_image.nii.gz'
        #change_orientation(or_ref_im,flo_im_file=Path(input[0]),flo_mask_file=Path(input[1]))

        flip_image(str(params.in_ct), str(proces_folder / "oriented_1.nii.gz"),axis=(1,2))
        flip_image(str(params.in_mask),str(proces_folder / "oriented_m_1.nii.gz"),axis=(1, 2))

        flip_image(str(params.in_ct), str(proces_folder / "oriented_2.nii.gz"),axis=(0,2))
        flip_image(str(params.in_mask),str(proces_folder / "oriented_m_2.nii.gz"),axis=(0, 2))

        flip_image(str(params.in_ct), str(proces_folder / "oriented_3.nii.gz"),axis=(0,1))
        flip_image(str(params.in_mask),str(proces_folder / "oriented_m_3.nii.gz"),axis=(0, 1))

        shell(f"cp -f \"{params.in_ct}\" \"{str(proces_folder / 'oriented_or.nii.gz')}\"")
        shell(f"cp -f \"{str(params.in_mask)}\" \"{str(proces_folder / 'oriented_m_or.nii.gz')}\"")

        try:
            register_flip(str(proces_folder / "oriented_3.nii.gz")
                ,reference_image=or_ref_im,
                float_mask=str(proces_folder / "oriented_m_3.nii.gz"),
                ref_mask=ref_mask_1,
                outputs=(str(proces_folder / "coreg_3.nii.gz"),str(proces_folder / "coreg_3.mat")))
        except:
            print("REGISTRATION FAILED ORIENTED_3")

        try:
            register_flip(str(proces_folder / "oriented_2.nii.gz")
                ,reference_image=or_ref_im,
                float_mask=str(proces_folder / "oriented_m_2.nii.gz"),
                ref_mask=ref_mask_1,
                outputs=(str(proces_folder / "coreg_2.nii.gz"),str(proces_folder / "coreg_2.mat")))
        except:
            print("REGISTRATION FAILED ORIENTED_2")
        try:
            register_flip(str(proces_folder / "oriented_1.nii.gz")
                ,reference_image=or_ref_im,
                float_mask=str(proces_folder / "oriented_m_1.nii.gz"),
                ref_mask=ref_mask_1,
                outputs=(str(proces_folder / "coreg_1.nii.gz"),str(proces_folder / "coreg_1.mat")))
        except:
            print("REGISTRATION FAILED ORIENTED_1")
        try:
            register_flip(str(proces_folder / "oriented_or.nii.gz")
                ,reference_image=or_ref_im,
                float_mask=str(proces_folder / "oriented_m_or.nii.gz"),
                ref_mask=ref_mask_1,
                outputs=(str(proces_folder / "coreg_or.nii.gz"),str(proces_folder / "coreg_or.mat")))
        except:
            print("REGISTRATION FAILED ORIENTED_or")
        ####check correlation and sort values


        subject_test = {}

        if (proces_folder/"coreg_or.nii.gz").exists():
            subject_test[str(proces_folder / "coreg_or.nii.gz")] = registration_quality(str(proces_folder / "coreg_or.nii.gz"),str(or_ref_im))
        if (proces_folder/"coreg_1.nii.gz").exists():
            subject_test[
                str(proces_folder / "coreg_1.nii.gz")] = registration_quality(str(proces_folder / "coreg_1.nii.gz"),str(or_ref_im))
        if (proces_folder / "coreg_2.nii.gz").exists():
            subject_test[
                str(proces_folder / "coreg_2.nii.gz")] = registration_quality(str(proces_folder / "coreg_2.nii.gz"),str(or_ref_im))
        if (proces_folder / "coreg_3.nii.gz").exists():
            subject_test[
                str(proces_folder / "coreg_3.nii.gz")] = registration_quality(str(proces_folder / "coreg_3.nii.gz"),str(or_ref_im))

        try:
            subject_test= list(sorted(subject_test.items(), key=lambda item: item[1], reverse=True))
            ###### copy to output
            final_coreg_im = subject_test[0][0]

            for i in range(1,len(subject_test)):
                Path(subject_test[i][0]).unlink()
            name = Path(final_coreg_im).name.split(".")[0]

            shell(f"mv -f '{final_coreg_im}' '{params.out_head}'")
            shell(f"mv -f '{str(proces_folder/(name + '.mat'))}' '{params.out_mat}'")

            ## remove temp files
            name_id = Path(subject_test[0][0]).name.split("_")[1]
            shell(f"mv -f '{str(proces_folder / ('oriented_m_' + name_id ))}' '{str(proces_folder / 'oriented_mask.nii.gz')}'")
            shell(f"mv -f '{str(proces_folder / ('oriented_' + name_id ))}' '{str(proces_folder / 'oriented_image.nii.gz')}'")

            for i in range(1,len(subject_test)):
                name = Path(subject_test[i][0]).name.split("_")[1]
                (proces_folder / (('oriented_') + name)).unlink()
                (proces_folder / (('oriented_m_') + name)).unlink()
        except :
            print("PROCESSING IS FAILED LOOK AT IMAGES!")

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
        shell(cmd)
        shell("rm -rf '" + def_field_file +"'" )





### rule for computing of mahalanobis pic's results

def read_coh_file():
    coh_file_l = open(coh_file,'rb')
    COHORTS = pickle.load(coh_file_l)

    coh_file_l.close()
    return COHORTS
    print("FILE READ")



rule general_procrustes:
    input: (OUT_FOLD/"{subject}/6_analyse_scalling.flag")
    output: (OUT_FOLD/"{subject}/6_procr_result.flag")
    params:
        in_ct = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/mahalanobis.mat",
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

            ##
            mat = np.loadtxt(params.in_matr)
            u,s,v = np.linalg.svd(mat)

            array = {"proc_dist": distance,
                     "abn_distance": "normal" if distance<980 else "abnormal",
                     "area99perc" : ratio, "abn_perc" : "normal" if ratio<0.1 else "abnormal","s_x": s[0],"s_y": s[1],"s_z":s[2]}
            df = pd.Series(array)
            df.to_csv(params.out_file)




rule combine_results:
    input: expand(OUT_FOLD/"{subject}/7_gifs.flag",subject = SRC_SUBJECTS)
    output: (PROCESS_DIR/"abnormality_automated_combined.csv")
    params:
        out_file = OUT_FOLD/"abnormality_automated_combined.csv"
    run:
        index = []
        res = []
        for fold in input:
            p_fold = Path(fold).parent
            index.append(p_fold.name)
            if (p_fold/ "abnormality_automated.csv").exists():
                subject =pd.read_csv((p_fold/ "abnormality_automated.csv"))
                subject.index = subject['Unnamed: 0']
                subject = subject.drop(columns=['Unnamed: 0'])
                subject = [x[0] for x in subject.values.tolist()]
                #subject[0] = float(subject[0])
                res.append(subject)
            else:
                res.append([None,None,None,None,None,None,None])
        print(res)
        pd.DataFrame(res,index=index,columns=["proc_dist","abn_distance",
                                              "area99perc", "abn_perc","s_x","s_y","s_z"]).to_csv(output[0])

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
        in_mat = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/affine_head.mat",
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

