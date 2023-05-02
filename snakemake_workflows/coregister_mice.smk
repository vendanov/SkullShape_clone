import pathlib

import SimpleITK as sitk
import pathlib
from scipy.ndimage.measurements import center_of_mass
import numpy as np


PROCESS_DIR = pathlib.Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir')
REF_IMAGE = pathlib.Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/reference_image/')

ref_im = REF_IMAGE / 'ct_image_.nii.gz'
ref_mask = REF_IMAGE / 'bone_coregistration_mask.nii.gz'

MICE_FOLDERS = [x for x in PROCESS_DIR.iterdir() if x.is_dir() and len(list(x.iterdir()))>0 ]

#MICE_FOLDERS = [x for x in PROCESS_DIR.iterdir() if x.is_dir() ]
#MICE_FOLDERS = [REF_IMAGE ]
#test mouse
#MICE_FOLDERS = [pathlib.Path(r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir/wt_M_82Z-CF113/") ]


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







def reorient_normal_image(image_orig :sitk.Image,label_orig:sitk.Image):
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(label_orig)

    image_result = sitk.Image(image_orig)
    image_result.SetDirection(label_shape_filter.GetOrientedBoundingBoxDirection(1))
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

########



rule all:
    input:
        expand("{dirname}/scalling_head.mat",dirname=MICE_FOLDERS)






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


rule linear_coregistration:
    input: "{dirname}/oriented_mask.nii.gz","{dirname}/oriented_image.nii.gz"
    output: "{dirname}/coregistered_head.nii.gz","{dirname}/affine_head.mat"
    threads: 1
    run:

        mask = pathlib.Path(input[0])
        flo_im = input[1]
        out_file = pathlib.Path(output[0])

        ref_mask_1 = REF_IMAGE/"oriented_mask.nii.gz"
        or_ref_im = REF_IMAGE / 'oriented_image.nii.gz'

        cmd = "reg_aladin -ref {refim} -rmask {rmask} -flo {flim} -fmask {fmask}".format(refim=or_ref_im,
            rmask=ref_mask_1,flim = input[1],fmask=input[0])
        cmd += " -speeeeed -cog -res " + str(out_file)
        cmd += " -aff " + output[1]

        shell(cmd)





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





