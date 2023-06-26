import pathlib
import subprocess
from typing import Union, Tuple
import vtk
import SimpleITK as sitk
import pathlib
from scipy.ndimage.measurements import center_of_mass
import numpy as np


PROCESS_DIR = pathlib.Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir')
REF_IMAGE = pathlib.Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/reference_image/')
PARAMETER_FILE = pathlib.Path("/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/Parameters_Stable_NL.txt")
ref_im = REF_IMAGE / 'ct_image_.nii.gz'
ref_mask = REF_IMAGE / 'bone_coregistration_mask.nii.gz'

#MICE_FOLDERS = [x for x in PROCESS_DIR.iterdir() if x.is_dir() and len(list(x.iterdir()))>0 ]

#MICE_FOLDERS = [x for x in PROCESS_DIR.iterdir() if x.is_dir() ]
#MICE_FOLDERS = [REF_IMAGE ]
#test mouse
#MICE_FOLDERS = [pathlib.Path(r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir/wt_M_82Z-CF113/") ]
MICE_FOLDERS = [pathlib.Path(r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir/wt_M_82Z-10911/"),
                pathlib.Path(r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir/wt_M_82Z-CF279/")]

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



    shell(cmd)
    return asses_registration_quality(outputs[0],reference_image)



def change_z_orientation( inp):
    inim = sitk.ReadImage(inp)
    direction = np.reshape(inim.GetDirection(),(3,3))
    direction[:,2] = direction[:,2] * -1
    direction[:, 0] = direction[:, 0] * -1
    inim.SetDirection(list(np.reshape(direction,-1)))
    sitk.WriteImage(inim,fileName=inp)



# def non_rigid_registration(flo_im: str, ref_im: str
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




def reverse_transform_mesh_points(mesh_points, inverse_transform):
    """

    :param mesh_points: points to convert
    :param inverse_transform:
    :return:
    """
    num_points = len(mesh_points)
    transformed_points = np.zeros((num_points, 3))

    for i in range(num_points):
        point = mesh_points[i]
        transformed_point = inverse_transform.TransformPoint(point)
        transformed_points[i] = transformed_point

    return transformed_points

def non_rigid_reverse_transformation(out_name: str,mesh_ref_space: str,
                                     deformation_field_file:str, linear_transform = None):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(mesh_ref_space)
    reader.Update()
    mesh = reader.GetOutput()
    points = mesh.GetPoints()

    # Get the number of points
    num_points = points.GetNumberOfPoints()

    # Create an empty list to store the coordinates
    point_list = []

    # Iterate over the points and retrieve their coordinates
    for i in range(num_points):
        coords = points.GetPoint(i)
        point_list.append(list(coords))

    # Print the resulting list of lists


    transform = sitk.ReadTransform(deformation_field_file)
    inverse_transform = sitk.DisplacementFieldTransform(transform.GetDisplacementField())
    inverse_transform = inverse_transform.GetInverse()

    points_native = reverse_transform_mesh_points(point_list,inverse_transform)

    # Create a new vtkPoints object and set the transformed points
    transformed_points = vtk.vtkPoints()
    for point in points_native:
        transformed_points.InsertNextPoint(point)

    # Create a new mesh with the transformed points
    mesh_transformed = vtk.vtkPolyData()
    mesh_transformed.SetPoints(transformed_points)
    mesh_transformed.SetPolys(mesh.GetPolys())

    # Write the transformed mesh to an OBJ file
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(out_name)
    writer.SetInputData(mesh_transformed)
    writer.Write()

########



rule all:
    input:
        expand("{dirname}/cpp_transformation.hdf5",dirname=MICE_FOLDERS)


rule otsu_threshold:
    input: "{dirname}/ct_image_.nii.gz"
    output: "{dirname}/bone_coregistration_mask.nii.gz"

    threads: 1

    run:
        inp_file = pathlib.Path(input[0])
        out_file = pathlib.Path(output[0])
        otsu_treshold(inp_file,out_file)


rule reorient_mask_and_im:
    input: "{dirname}/bone_coregistration_mask.nii.gz"
    output: "{dirname}/oriented_mask.nii.gz","{dirname}/oriented_image.nii.gz"
    threads: 1
    run:

        mask = pathlib.Path(input[0])
        flo_im = mask.parent / "ct_image_.nii.gz"

        mask_im = sitk.ReadImage(mask)
        ## reorient main image
        image1_res = reorient_normal_image(sitk.ReadImage(flo_im),mask_im)
        sitk.WriteImage(image1_res,output[1])
        ###reorient mask
        image1_res = reorient_normal_image(mask_im,mask_im)
        sitk.WriteImage(image1_res,output[0])
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

rule linear_coregistration:
    input: "{dirname}/oriented_mask.nii.gz","{dirname}/oriented_image.nii.gz"
    output: "{dirname}/coregistered_head.nii.gz","{dirname}/affine_head.mat"
    threads: 1
    run:

        out_file = pathlib.Path(output[0])

        ref_mask_1 = REF_IMAGE/"oriented_mask.nii.gz"
        or_ref_im = REF_IMAGE / 'oriented_image.nii.gz'

        lcc = register_flip(input[1],reference_image=or_ref_im,
            float_mask=input[0],ref_mask=ref_mask_1,outputs=output)
        if lcc < -20:
            change_z_orientation(input[0])
            change_z_orientation(input[1])

            print(lcc)
            lcc = register_flip(input[1],reference_image=or_ref_im,
                float_mask=input[0],
                ref_mask=ref_mask_1,outputs=output)



rule non_rigid_registration_rule:
    input: "{dirname}/coregistered_head.nii.gz"
    output: "{dirname}/nonlin_image.nii.gz","{dirname}/cpp_transformation.hdf5"
    threads: 1
    run:
        or_ref_im = REF_IMAGE / 'average_image.nii.gz'
        non_rigid_registration(flo_im=input[0], ref_im=str(or_ref_im),
            out_transform=output[1],out_image=output[0],param_file=str(PARAMETER_FILE))

rule inverse_transform_mesh_rule:
    input: "{dirname}/cpp_transformation.hdf5"
    output: "{dirname}/head_affine_registered.obj",
    threads: 1
    run:
        or_ref_im = REF_IMAGE / "smooth_low.obj"
        non_rigid_reverse_transformation(out_name=output[0]
            ,mesh_ref_space=str(or_ref_im),deformation_field_file=input[0])





