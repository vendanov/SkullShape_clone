import os
import pathlib
import subprocess
from pathlib import Path
from typing import Union, Tuple

import SimpleITK as sitk
import numpy as np
import vtk
import shlex

PROCESS_DIR = Path(config['process_dir'])
print(PROCESS_DIR)
#pathlib.Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir')
REF_IMAGE = pathlib.Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/reference_image/')
PARAMETER_FILE_NONLIN = pathlib.Path("/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/Parameters_Stable_NL.txt")
PARAMETER_FILE_LIN = pathlib.Path("/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/scripts/models/affine.txt")
ref_im = REF_IMAGE / 'ct_image_.nii.gz'
ref_mask = REF_IMAGE / 'bone_coregistration_mask.nii.gz'
MESH_TRANSFORM_SCRIPT = "/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/scripts/support/convert_mesh.py"
SLICER_CORE = "/home/varhai/Slicer-5.2.2-linux-amd64/Slicer"


MICE_FOLDERS = [x for x in PROCESS_DIR.iterdir() if x.is_dir() and len(list(x.iterdir()))>0 ]

#MICE_FOLDERS = [x for x in PROCESS_DIR.iterdir() if x.is_dir() ]
#MICE_FOLDERS = [REF_IMAGE ]
#test mouse
#MICE_FOLDERS = [pathlib.Path(r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir/wt_M_82Z-CG443/") ]
# MICE_FOLDERS = [pathlib.Path(r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir/wt_F_82Z-11537/"),
#                 pathlib.Path(r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir/wt_F_82Z-09866/")]
### testfor new linreg
 # MICE_FOLDERS = [pathlib.Path(r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_ablormal/Cog5_F_E64-G2766/"),
 #                 pathlib.Path(r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_ablormal/Cog5_M_E64-G2762/")]
def otsu_treshold(input_file: pathlib.Path, out_file: pathlib.Path):
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



def get_frontal_voxels(image:sitk.Image,front=True):
    direction = image.GetDirection()
    direction = np.reshape(direction,(3,3))
    origin = image.GetOrigin()
    inverse_direction = np.linalg.inv(direction)
    size = image.GetSize()
    frontal_voxels = np.zeros(size, dtype=np.uint8)
    if front:
        cmp = lambda x,y : x < y
    else:
        cmp = lambda x,y : x > y
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                position = np.dot(inverse_direction, (x, y, z)) + origin

                if cmp(position[2],origin[2]):
                    frontal_voxels[x, y, z] = 1

    return frontal_voxels.sum()

def reorient_normal_image(image_orig :sitk.Image,label_orig:sitk.Image):
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(label_orig)

    image_result = sitk.Image(image_orig)
    direction = label_shape_filter.GetOrientedBoundingBoxDirection(1)

    # if get_frontal_voxels(label_orig) > get_frontal_voxels(label_orig,False):
    #     print(True)
    # return
    image_result.SetDirection(direction)
    image_result.SetOrigin(label_shape_filter.GetCentroid(1))

    return image_result



def coreg_mask(ref,flo,out):
    # Load the images
    image1 = sitk.ReadImage(flo)

    image1_res = reorient_normal_image(image1,image1)


    sitk.WriteImage(image1_res,out)

def to_rigid_transf(matrix,out,scale_out):
    affine_matrix = np.loadtxt(matrix)

    # Perform SVD decomposition
    U, s, V = np.linalg.svd(affine_matrix[:3, :3])

    # Extract the rotation and translation matrices
    rotation_matrix = U @ V
    translation_matrix = affine_matrix[:3, 3]

    res = np.eye(4)

    res[:3, :3] = rotation_matrix
    res[:3, 3] = affine_matrix[:3, 3]


    np.savetxt(out,res)

    res = np.eye(4)
    res[:3, :3]= np.diag(s)
    np.savetxt(scale_out,res)
    # Print the rigid component
    print('Rotation Matrix:\n',rotation_matrix)
    print('Translation Matrix:\n',translation_matrix)
    print("Scalling: \n",s)


def asses_registration_quality(registered_image: Union[str,pathlib.Path], reference_image: Union[str,pathlib.Path]):

    #niftyreg_cmd = "reg_measure -ref {rim} -flo {flim} -ssd".format(rim = reference_image,flim=registered_image)
    niftyreg_cmd = ['reg_measure',"-ref", reference_image,"-flo",registered_image,"-ssd"]
    # Call NiftyReg using subprocess
    niftyreg_output = subprocess.check_output(niftyreg_cmd)

    # Extract the NMI value from the NiftyReg output

    ssd = float(niftyreg_output.decode("utf-8").split(" ")[1])
    print(ssd)
    return ssd


# def register_affine(ref_image,flo_im,output_folder,out_image):
#     cmd =  f"elastix -f {str(ref_image)} -m {str(flo_im)} -p {str(PARAMETER_FILE_LIN)} " \
#            # f"-fMask {str(ref_image.parent / 'oriented_mask.nii.gz')} " \
#            # f"-mMask {str(flo_im.parent / 'oriented_mask.nii.gz')} " \
#            f"-out {str(output_folder)}"
#     shell(cmd)
#
#     shell(f"mv -f {str(output_folder/ 'result.0.nii.gz')} {str(out_image)}")
#
#     transform_parameters = sitk.ReadParameterFile(str(output_folder/ 'TransformParameters.0.txt') )
#     linear_parameters = []
#     for param in transform_parameters:
#         if 'Transform' in param[0]:
#             linear_parameters.extend(list(map(float, param[1:])))
#
#     # Extract the affine transformation matrix from the linear parameters
#     affine_matrix = np.array([[linear_parameters[0], linear_parameters[1], linear_parameters[2], linear_parameters[3]],
#                               [linear_parameters[4], linear_parameters[5], linear_parameters[6], linear_parameters[7]],
#                               [linear_parameters[8], linear_parameters[9], linear_parameters[10], linear_parameters[11]],
#                               [0.0, 0.0, 0.0, 1.0]])
#     #write transformation
#     res = ""
#     for i in range(affine_matrix.shape[0]):
#         res += " ".join([str(x) for x in affine_matrix[i].tolist()]) + "\n"
#
#     with open(str(output_folder/ "affine_head.mat"),"wt") as f:
#         f.write(res)

# def register_affine(ref_image,flo_im,output_folder,out_image):
#     fixed_image = sitk.ReadImage(ref_image,sitk.sitkFloat32)
#     moving_image = sitk.ReadImage(flo_im,sitk.sitkFloat32)
#     initial_transform = sitk.CenteredTransformInitializer(fixed_image,
#         moving_image,
#         sitk.Euler3DTransform(),
#         sitk.CenteredTransformInitializerFilter.MOMENTS)
#
#     registration_method = sitk.ImageRegistrationMethod()
#     registration_method.SetMetricAsCorrelation()
#     # registration_method.SetInterpolator(sitk.sitkLinear) #2. Replace with sitkLinear
#     registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
#         numberOfIterations=20,
#         maximumNumberOfCorrections=5,
#         maximumNumberOfFunctionEvaluations=1000,
#         costFunctionConvergenceFactor=1e+7)  #1. Increase to 1000
#
#     # Don't optimize in-place, we would like to run this cell multiple times
#     registration_method.SetInitialTransform(initial_transform,inPlace=False)
#
#




def register_flip(floating_image: Union[str,pathlib.Path],
                  reference_image: Union[str,pathlib.Path], float_mask:Union[str,pathlib.Path],
                  ref_mask :Union[str,pathlib.Path],
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
        cmd = "reg_aladin -ref {refim} -rmask {rmask} -flo {flim} -fmask {fmask}".format(refim=reference_image,
            rmask=ref_mask,flim=floating_image,fmask=float_mask)
        cmd += " -speeeeed -cog -res " + str(outputs[0])
        cmd += " -aff " + outputs[1]

    else:
        cmd = "reg_aladin -ref {refim} -rmask {rmask} -flo {flim} -fmask {fmask}".format(refim=reference_image,
            rmask=ref_mask,flim=floating_image,fmask=float_mask)
        cmd += " -speeeeed -cog -res " + str(outputs[0])
        cmd += " -aff " + str(outputs[1]) + " -inaff " + str(inaff)


    subprocess.call(shlex.split(cmd))

    #shell(cmd)
    #return asses_registration_quality(outputs[0],reference_image)


def generate_orienter(fixed_image,moving_image):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()


    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetMetricAsCorrelation()

    registration_method.SetOptimizerAsExhaustive(numberOfSteps=[4, 4, 0, 0, 0, 0],stepLength=np.pi / 4)
    registration_method.SetOptimizerScales([1, 1, 1, 1, 1, 1])
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform)
    return registration_method

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

def change_orientation( refim_file, flo_im_file, flo_mask_file=None):
    fixed_image = sitk.ReadImage(refim_file,sitk.sitkFloat32)
    moving_image = sitk.ReadImage(flo_im_file,sitk.sitkFloat32)



    registration_method = generate_orienter(fixed_image,moving_image)

    # Connect all of the observers so that we can perform plotting during registration.

    final_transform = registration_method.Execute(fixed_image,moving_image)



    # im  resample
    moving_resampled = sitk.Resample(moving_image,fixed_image,final_transform,sitk.sitkNearestNeighbor,0.0,moving_image.GetPixelID())
    sitk.WriteImage(moving_resampled,flo_im_file)
    # mask resample
    if flo_mask_file is not None:
        flo_mask = sitk.ReadImage(flo_mask_file,sitk.sitkUInt8)
        print(flo_mask_file)
        moving_resampled = sitk.Resample(flo_mask,fixed_image,final_transform,sitk.sitkNearestNeighbor,0.0,flo_mask.GetPixelID())
        sitk.WriteImage(moving_resampled,flo_mask_file)



# def non_rigid_registration(flo_im: str, ref_im: strcd
#                            , out_transform:str, out_image: str):
#
#     cmd = f"reg_f3d -ref {ref_im} -flo {flo_im} -cpp {out_transform} -res {out_image}"
#     shell(cmd)


def non_rigid_registration(flo_im: str, ref_im: str, out_transform: str, out_image: str, param_file:str):
    # Reading
    fold = pathlib.Path(out_image).parent
    cmd = f"elastix -f {ref_im} -m {flo_im} -p {param_file} -out {str(fold)}"
    shell(cmd)
    shell(f"mv -f {str(fold/ 'result.0.nii.gz')} {out_image}")
    shell(f"mv -f {str(fold/ 'TransformParameters.0.txt')} {out_transform}")




def transform_mesh_points(mesh_points, transform):
    """

    :param mesh_points: points to convert
    :param transform:
    :return:
    """
    num_points = len(mesh_points)
    transformed_points = np.zeros((num_points, 3))

    for i in range(num_points):
        point = mesh_points[i]
        transformed_point = transform.TransformPoint(point)
        transformed_points[i] = transformed_point

    return transformed_points

def read_output_points(filename: str) -> np.ndarray:

    with open(filename,"rt") as file:
        lines = file.readlines()
        points = []

        #print(len(lines))
        for l in lines:
            #print(l.split(" OutputPoint = [ ")[1].split("]")[0].split())
            point = [float(x) for x in l.split(" OutputPoint = [ ")[1].split("]")[0].split()]
            points.append(point)
    return np.array(points)




def replace_points_in_mesh(out_name: str, mesh_ref_space: str,
                                     transformed_poits_file:str, linear_transform = None):
    """
    replaces points to one produced in transormix and save meshe in target file
    :param out_name:
    :param mesh_ref_space:
    :param transformed_poits_file:
    :param linear_transform:
    :return:
    """
    reader = vtk.vtkOBJReader()
    reader.SetFileName(mesh_ref_space)
    reader.Update()
    mesh = reader.GetOutput()
    points = mesh.GetPoints()

    # Get the number of points
    num_points = points.GetNumberOfPoints()

    transformed_points = read_output_points(transformed_poits_file)

    res_pts = vtk.vtkPoints()
    for pt in transformed_points.tolist():
        res_pts.InsertNextPoint(pt)

    mesh_transformed = vtk.vtkPolyData()
    mesh_transformed.SetPoints(res_pts)
    mesh_transformed.SetPolys(mesh.GetPolys())

    # Write the transformed mesh to an OBJ file
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(out_name)
    writer.SetInputData(mesh_transformed)
    writer.Write()

########



rule all:
    input:
        expand("{dirname}/head_affine_registered.obj",dirname=MICE_FOLDERS)


rule otsu_threshold:
    input: "{dirname}/ct_image_.nii.gz"
    output: "{dirname}/bone_coregistration_mask.nii.gz"

    threads: 1

    run:
        inp_file = pathlib.Path(input[0])
        out_file = pathlib.Path(output[0])

        otsu_treshold(inp_file,out_file)


def reflip_image( image_file):
    moving_image = sitk.ReadImage(fileName=image_file)
    direction = np.reshape(moving_image.GetDirection(),(3,3))
    direction[:,2] = direction[:,2] * -1
    direction[:, 0] = direction[:, 0] * -1

    moving_image.SetDirection(list(np.reshape(direction,-1)))
    sitk.WriteImage(moving_image,image_file)


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


rule reorient_mask_and_im:
    input: "{dirname}/bone_coregistration_mask.nii.gz"
    output: "{dirname}/oriented_mask.nii.gz","{dirname}/oriented_image.nii.gz"
    threads: 1
    run:

        mask = pathlib.Path(input[0])
        flo_im = mask.parent / "ct_image_.nii.gz"

        mask_im = sitk.ReadImage(mask)
        or_ref_im = REF_IMAGE / "oriented_image.nii.gz"

        shell(f"cp -f {str(flo_im)} {output[1]}")
        shell(f"cp -f {input[0]} {output[0]}")
        change_orientation(or_ref_im,flo_im_file=output[1],flo_mask_file=Path(output[0]))


        ## reorient main image
        #image1_res = reorient_normal_image(sitk.ReadImage(flo_im),mask_im)
        #sitk.WriteImage(image1_res,output[1])
        ###reorient mask
        #image1_res = reorient_normal_image(mask_im,mask_im)
        #sitk.WriteImage(image1_res,output[0])
        #shell(cmd)





rule to_rigid:
    input: "{dirname}/affine_head.mat"
    output: "{dirname}/rigid_head.mat", "{dirname}/scalling_head.mat"
    threads: 1
    run:
        to_rigid_transf(input[0],output[0],output[1])

rule whole_head_rigid:
    input: "{dirname}/affine_head.mat", "{dirname}/ct_image_.nii.gz"
    output: "{dirname}/rigid_head.nii.gz",
    threads: 1
    run:
        or_ref_im = REF_IMAGE / 'oriented_image.nii.gz'

        cmd = "reg_resample -ref {refim} -flo {flim} -trans {trmat} -res {outmat}".format(refim=or_ref_im,
            flim=input[1],trmat=input[0],outmat = output[0])

# rule linear_coregistration:
#     input: "{dirname}/oriented_mask.nii.gz","{dirname}/oriented_image.nii.gz"
#     output: "{dirname}/coregistered_head.nii.gz","{dirname}/affine_head.mat"
#     threads: 2
#     run:
#
#         out_file = pathlib.Path(output[0])
#
#         ref_mask_1 = REF_IMAGE/"oriented_mask.nii.gz"
#         or_ref_im = REF_IMAGE / 'oriented_image.nii.gz'
#
#         register_affine(ref_image=or_ref_im, flo_im = pathlib.Path(input[1]),output_folder = out_file.parent,out_image=output[0])


rule linear_coregistration:
    input: "{dirname}/ct_image_.nii.gz","{dirname}/bone_coregistration_mask.nii.gz"
    output: "{dirname}/coregistered_head.nii.gz","{dirname}/affine_head.mat"
    threads: 2
    run:
        proces_folder = Path(input[0]).parent
        out_file = Path(output[0])

        ref_mask_1 = REF_IMAGE/"oriented_mask.nii.gz"
        or_ref_im = REF_IMAGE / 'oriented_image.nii.gz'
        #change_orientation(or_ref_im,flo_im_file=Path(input[0]),flo_mask_file=Path(input[1]))

        flip_image(input[0], str(proces_folder / "oriented_1.nii.gz"),axis=(1,2))
        flip_image(input[1],str(proces_folder / "oriented_m_1.nii.gz"),axis=(1, 2))

        flip_image(input[0], str(proces_folder / "oriented_2.nii.gz"),axis=(0,2))
        flip_image(input[1],str(proces_folder / "oriented_m_2.nii.gz"),axis=(0, 2))

        flip_image(input[0], str(proces_folder / "oriented_3.nii.gz"),axis=(0,1))
        flip_image(input[1],str(proces_folder / "oriented_m_3.nii.gz"),axis=(0, 1))

        shell(f"cp -f {input[0]} {str(proces_folder / 'oriented_or.nii.gz')}")
        shell(f"cp -f {input[1]} {str(proces_folder / 'oriented_m_or.nii.gz')}")


        register_flip(str(proces_folder / "oriented_3.nii.gz")
            ,reference_image=or_ref_im,
            float_mask=str(proces_folder / "oriented_m_3.nii.gz"),
            ref_mask=ref_mask_1,
            outputs=(str(proces_folder / "coreg_3.nii.gz"),str(proces_folder / "coreg_3.mat")))

        register_flip(str(proces_folder / "oriented_2.nii.gz")
            ,reference_image=or_ref_im,
            float_mask=str(proces_folder / "oriented_m_2.nii.gz"),
            ref_mask=ref_mask_1,
            outputs=(str(proces_folder / "coreg_2.nii.gz"),str(proces_folder / "coreg_2.mat")))

        register_flip(str(proces_folder / "oriented_1.nii.gz")
            ,reference_image=or_ref_im,
            float_mask=str(proces_folder / "oriented_m_1.nii.gz"),
            ref_mask=ref_mask_1,
            outputs=(str(proces_folder / "coreg_1.nii.gz"),str(proces_folder / "coreg_1.mat")))

        register_flip(str(proces_folder / "oriented_or.nii.gz")
            ,reference_image=or_ref_im,
            float_mask=str(proces_folder / "oriented_m_or.nii.gz"),
            ref_mask=ref_mask_1,
            outputs=(str(proces_folder / "coreg_or.nii.gz"),str(proces_folder / "coreg_or.mat")))
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

        subject_test= list(sorted(subject_test.items(), key=lambda item: item[1], reverse=True))
        ###### copy to output
        final_coreg_im = subject_test[0][0]

        name = Path(final_coreg_im).name.split(".")[0]

        shell(f"mv -f {final_coreg_im} {output[0]}")
        shell(f"mv -f {str(proces_folder/(name + '.mat'))} {output[1]}")

        ## remove temp files
        name_id = Path(subject_test[0][0]).name.split("_")[1]
        shell(f"rm -f {str(proces_folder / ('oriented_m_' + name_id + '.nii.gz'))}")
        shell(f"rm -f {str(proces_folder / ('oriented_' + name_id + '.nii.gz'))}")

        shell(f"rm -rf {str(proces_folder / ('oriented_' ))}*")
        shell(f"rm -rf {str(proces_folder / ('coreg_'))}*")
        # for i_sub in range(1,len(subject_test)):
        #     shell(f"rm -f {subject_test[i_sub][0]}")
        #     shell(f"rm -f {subject_test[i_sub][0].split('.')[0] + '.mat'}")
        #     name_id = Path(subject_test[i_sub][0]).name.split("_")[1]
        #     shell(f"rm -f {str(proces_folder / ('oriented_m_'+ name_id +'.nii.gz'))}")
        #     shell(f"rm -f {str(proces_folder / ('oriented_'+ name_id +'.nii.gz'))}")



#


rule non_rigid_registration_rule:
    input: "{dirname}/coregistered_head.nii.gz"
    output: "{dirname}/nonlin_image.nii.gz","{dirname}/nl_transformation_params.txt"
    threads: 2
    run:
        # script coregisters reference image (choosen as default to each linearly registered head for reverse mesh generation
        or_ref_im = REF_IMAGE / 'oriented_image.nii.gz'
        non_rigid_registration(flo_im=str(or_ref_im), ref_im=input[0],
            out_transform=output[1],out_image=output[0],param_file=str(PARAMETER_FILE_NONLIN))




rule inverse_transform_mesh_rule:
    input: "{dirname}/nl_transformation_params.txt"
    output: "{dirname}/head_affine_registered.obj",
    threads: 2
    run:
        dir1name = str(pathlib.Path(input[0]).parent)

        #generate deformation field
        cmd = f"transformix -tp {input[0]} -out {dir1name} -def all"

        shell (cmd)
        or_ref_mesh = REF_IMAGE / "oriented_mesh.obj"
        print(or_ref_mesh)

        def_field_file  = dir1name + os.sep + "deformationField.nii.gz"
        ###deform original mesh

        cmd = f"xvfb-run -a  {SLICER_CORE} --python-script {MESH_TRANSFORM_SCRIPT} {def_field_file} {or_ref_mesh} {output[0]}  --testing --no-splash --no-main-window"
        shell(cmd)
        shell("rm -rf " + def_field_file )




