import shutil
import sys
from pathlib import Path
import subprocess

import numpy as np
import SimpleITK as sitk

from mouse_skull_pipelines.scripts.image_algorithms import register_linear_full, non_rigid_registration, \
    LinearRegistrationConfig, otsu_treshold

# as input we need the following structure:
# cohort_folder/{subfolder}/ct.nii.gz
# as reference image we need the following structure:
# ref_folder/ref_image.nii.gz ref_folder/ref_mesh.obj


# the output will be:
# out_folder/{subfolder}/res_mesh.obj out_folder/{subfolder}/res_affine.mat

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
ct_im_name = "ct_image.nii.gz"

print("PROCESS_DIR", PROCESS_DIR)

CONFIG_SUBJECTS = config.get("subjects") if "subjects" in config else None
SRC_SUBJECTS = [fold.name for fold in PROCESS_DIR.iterdir()
                if (fold.is_dir() and (fold / ct_im_name).exists())]
if CONFIG_SUBJECTS:
    SRC_SUBJECTS = [s for s in SRC_SUBJECTS if s in set(CONFIG_SUBJECTS)]


print("src_subects" ,SRC_SUBJECTS)

PROCESS_d = str(PROCESS_DIR)
rule all:
    input:
        expand(PROCESS_d +"/{cf}/4_inv_mesh.flag",cf = SRC_SUBJECTS)



rule otsu_threshold:
    input:
        PROCESS_DIR / "{subject}"/ ct_im_name
    output:
        flag=(PROCESS_DIR / "{subject}/1_otsu.flag")
    threads: 1

    run:
        inp_file = Path(output.flag).parent / ct_im_name
        out_file = Path(output.flag).parent / "bone_coregistration_mask.nii.gz"
        out_file.unlink(missing_ok=True)



        otsu_treshold(inp_file,out_file)

        Path(output.flag).touch(exist_ok=True)



def flip_image_and_correct_affine(input_path: str, output_path: str):
    # Чтение изображения
    img = sitk.ReadImage(input_path)
    arr = sitk.GetArrayFromImage(img)  # (z,y,x) порядок в sitk
    spacing = img.GetSpacing()
    origin = list(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3,3)

    # Флип по x (axis=2) и y (axis=1)
    flipped_arr = np.flip(arr, axis=(1,2))  # flip y and x

    # Меняем Origin
    size = img.GetSize()
    origin[0] = origin[0] + spacing[0] * (size[0] - 1)
    origin[1] = origin[1] + spacing[1] * (size[1] - 1)

    # Меняем Direction (разворачиваем оси)
    direction[0,0] *= -1
    direction[1,1] *= -1

    # Создаем новое изображение
    flipped_img = sitk.GetImageFromArray(flipped_arr)
    flipped_img.SetSpacing(spacing)
    flipped_img.SetOrigin(origin)
    flipped_img.SetDirection(direction.flatten())

    # Сохраняем
    sitk.WriteImage(flipped_img, output_path)

def hard_flip(input_path: str, output_path: str):
    img = sitk.ReadImage(input_path)
    arr = sitk.GetArrayFromImage(img)  # (z,y,x)

    # Просто флип данных (без изменения метаданных)
    flipped_arr = np.flip(arr, axis=(1, 2))  # flip y and x

    # Создать новое изображение с теми же origin, spacing, direction
    flipped_img = sitk.GetImageFromArray(flipped_arr)
    flipped_img.SetSpacing(img.GetSpacing())
    flipped_img.SetOrigin(img.GetOrigin())
    flipped_img.SetDirection(img.GetDirection())

    sitk.WriteImage(flipped_img, output_path)


def flip_image(input_path: str, output_path: str, axes=(False, False, False), correct_affine=True):
    """
    Flip an image along specified axes.

    Args:
        input_path: Path to input .nii/.mha/.nrrd image.
        output_path: Where to save flipped image.
        axes: Tuple of booleans (flip_x, flip_y, flip_z), axis in physical space.
        correct_affine: If True, updates origin and direction accordingly.
    """
    img = sitk.ReadImage(input_path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)

    flip_axes = [i for i, flip in enumerate(axes[::-1]) if flip]  # numpy axes (x=2, y=1, z=0)

    # Flip voxel data
    flipped_arr = np.flip(arr,axis=flip_axes) if flip_axes else arr.copy()

    # Create new image
    flipped_img = sitk.GetImageFromArray(flipped_arr)
    flipped_img.SetSpacing(img.GetSpacing())

    if correct_affine:
        spacing = img.GetSpacing()
        origin = list(img.GetOrigin())
        direction = np.array(img.GetDirection()).reshape(3,3)
        size = img.GetSize()

        for i, flip in enumerate(axes):  # i: 0=z, 1=y, 2=x
            if flip:
                origin[i] += spacing[i] * (size[i] - 1)
                direction[:, i] *= -1

        flipped_img.SetOrigin(origin)
        flipped_img.SetDirection(direction.flatten())
    else:
        flipped_img.SetOrigin(img.GetOrigin())
        flipped_img.SetDirection(img.GetDirection())

    sitk.WriteImage(flipped_img,output_path)


rule linear_coregistration:
    input: PROCESS_DIR / "{subject}"/ ct_im_name,
        (PROCESS_DIR / "{subject}/1_otsu.flag")

    output: PROCESS_DIR / "{subject}/2_lin_cor.flag"
    params:
        in_landmarks=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/landmarks.pts",
        in_mask=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/bone_coregistration_mask.nii.gz",
        out_head=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/coregistered_head.nii.gz",
        out_mat=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/rigid_head.txt",
        out_affine=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/affine_head.txt"

    threads: 1
    run:
        process_folder = Path(output[0]).parent
        Path(str(params.out_head)).unlink(missing_ok=True)
        Path(str(params.out_mat)).unlink(missing_ok=True)
        Path(str(params.out_affine)).unlink(missing_ok=True)

        in_ct = Path(input[0])
        out_mask = process_folder / "coregistered_mask.nii.gz"


        config : LinearRegistrationConfig  = LinearRegistrationConfig(is_simple=LINEAR_SIMPLE,
            ref_landmarks=None,
            side_landmarks_model=None,control_points_model=None)
        metric = register_linear_full(str(in_ct),str(ref_im),
            str(params.in_mask),ref_mask=str(ref_mask),
            config=config,


            outputs=(str(params.out_head), str(process_folder / 'TransformParameters.0.txt')))
        print("metric!",metric)
        if metric < 0.5:
            print("linear registration failed")
            flip_image(str(in_ct), str(params.out_head), axes=(False, True, True), correct_affine=False)
            flip_image(str(params.in_mask), str(out_mask), axes=(False, True, True), correct_affine=False)
            metric = register_linear_full(str(params.out_head),str(ref_im),
                str(out_mask),ref_mask=str(ref_mask),
                config=config,
                outputs=(str(params.out_head), str(process_folder / 'TransformParameters.0.txt')))
            print("metric_3",metric)
        if metric < 0.5:
            # flip image and mask
            hard_flip(str(in_ct), str(params.out_head))
            hard_flip(str(params.in_mask), str(out_mask))
            metric = register_linear_full(str(params.out_head),str(ref_im),
                str(out_mask),ref_mask=str(ref_mask),
                config=config,
                outputs=(str(params.out_head), str(process_folder / 'TransformParameters.0.txt')))

            print("metric_2",metric)



        aff_mat = np.loadtxt(process_folder / 'rigid_head.txt')
        # read the affine matrix from the file
        affine_matrix = np.loadtxt(process_folder / 'TransformParameters.0.txt')

        full_head_matrix = affine_matrix @ aff_mat

        np.savetxt(str(params.out_affine),full_head_matrix)
        # Convert LPS (SimpleITK) -> RAS (Slicer) so the .h5 applies correctly in Slicer.
        lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
        ras_matrix = lps_to_ras @ full_head_matrix @ lps_to_ras
        affine_tx = sitk.AffineTransform(3)
        affine_tx.SetMatrix(ras_matrix[:3, :3].reshape(-1).tolist())
        affine_tx.SetTranslation(ras_matrix[:3, 3].tolist())
        sitk.WriteTransform(affine_tx, str(process_folder / "affine_head.h5"))
        #print("difference after affine transform",fix_pts_np - transform_points(mov_pts_np,full_head_matrix))
        #

        fn1 = process_folder / 'TransformParameters.0.txt'
        shell(f'rm -f "{str(fn1)}"')

        #raise ValueError("STOP")
        Path(output[0]).touch(exist_ok=True)



rule non_rigid_registration_rule:
    input:
        PROCESS_DIR / "{subject}/2_lin_cor.flag"
    output: PROCESS_DIR / "{subject}/3_ants_nonrig.flag"
    params:
        inp_coregim=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/coregistered_head.nii.gz",
        warped=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/warped_elastic.nii.gz",
        deformation=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/deformationField.h5"
    threads: 2
    run:
        warped_path = Path(str(params.warped))
        deformation_path = Path(str(params.deformation))
        warped_path.unlink(missing_ok=True)
        deformation_path.unlink(missing_ok=True)
        import ants

        fixed = ants.image_read(str(params.inp_coregim))
        moving = ants.image_read(str(REF_IMAGE / 'oriented_image.nii.gz'))

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="SyNOnly",# flexible non-symmetric deformation
            grad_step=0.5,
            flow_sigma=1,
            total_sigma=0,
            singleprecision=True,
            write_composite_transform=True,
            verbose=True
        )

        fwd_transforms = reg["fwdtransforms"]
        print("src_transform", fwd_transforms)

        warped = reg["warpedmovout"]
        ants.image_write(warped, str(warped_path))

        composite_transform = fwd_transforms[0] if isinstance(fwd_transforms, (list, tuple)) else fwd_transforms
        shutil.copy(composite_transform, str(deformation_path))

        inv_transforms = reg['invtransforms']
        if isinstance(inv_transforms, str):
            Path(inv_transforms).unlink(missing_ok=True)
        else:
            for transform_path in inv_transforms:
                Path(transform_path).unlink(missing_ok=True)
        if isinstance(fwd_transforms, str):
            Path(fwd_transforms).unlink(missing_ok=True)
        else:
            for transform_path in fwd_transforms:
                Path(transform_path).unlink(missing_ok=True)

        Path(str(output[0])).touch(exist_ok=True)



rule elastix_refinement_rule:
    input:
        ants_flag=PROCESS_DIR / "{subject}/3_ants_nonrig.flag",
    output: PROCESS_DIR / "{subject}/3_nonrig.flag"
    params:
        warped=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/warped_elastic.nii.gz",
        fixed=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/coregistered_head.nii.gz",
        refined_image=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/nonlin_image.nii.gz",
        transform_params=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/nl_transformation_params.txt",
        deformation_field=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/elastix_deformation_field.h5"
    threads: 2
    run:
        refined_image_path = Path(str(params.refined_image))
        refined_image_path.unlink(missing_ok=True)
        transform_path = Path(str(params.transform_params))
        if transform_path.exists():
            transform_path.unlink()
        for existing_file in transform_path.parent.glob(f"{transform_path.stem}_*{transform_path.suffix}"):
            existing_file.unlink(missing_ok=True)
        deformation_field_path = Path(str(params.deformation_field))
        deformation_field_path.unlink(missing_ok=True)
 
        elastix = sitk.ElastixImageFilter()
        elastix.SetLogToConsole(False)
        elastix.SetLogToFile(False)
        sitk_fixed = sitk.ReadImage(str(params.fixed))
        sitk_moving_initial = sitk.ReadImage(str(params.warped))
        elastix.SetFixedImage(sitk_fixed)
        elastix.SetMovingImage(sitk_moving_initial)

        if PARAMETER_FILE_NONLIN.exists():
            parameter_map = sitk.ReadParameterFile(str(PARAMETER_FILE_NONLIN))
        else:
            parameter_map = sitk.GetDefaultParameterMap("bspline")
        elastix.SetParameterMap(parameter_map)
        elastix.Execute()

        refined_image = elastix.GetResultImage()
        sitk.WriteImage(refined_image, str(refined_image_path))

        transform_maps = elastix.GetTransformParameterMap()
        for idx, parameter_map in enumerate(transform_maps):
            if idx == 0:
                target_path = transform_path
            else:
                target_path = transform_path.parent / f"{transform_path.stem}_{idx}{transform_path.suffix}"
            sitk.WriteParameterFile(parameter_map, str(target_path))

            transformix = sitk.TransformixImageFilter()
            transformix.SetTransformParameterMap(transform_maps)
            # You don't need the moving image to compute the deformation field:
            # transformix.SetMovingImage(sitk_moving_initial)   # optional; only needed if you also resample the moving image
            transformix.SetComputeDeformationField(True)        # equivalent to ComputeDeformationFieldOn()
            transformix.SetMovingImage(sitk_moving_initial)  # Required to set size/spacing/origin/direction of the deformation field
            transformix.Execute()

            deformation_field = transformix.GetDeformationField()

            # --- FIX: cast to VectorFloat64 for DisplacementFieldTransform ---
            if deformation_field.GetPixelID() != sitk.sitkVectorFloat64:
                deformation_field = sitk.Cast(deformation_field, sitk.sitkVectorFloat64)

            # (optional sanity checks; helps avoid silent metadata mismatches)
            assert deformation_field.GetNumberOfComponentsPerPixel() == 3, "Deformation field must have 3 components."
            for a, b, name in [
                (deformation_field.GetSize(),      sitk_fixed.GetSize(),      "size"),
                (deformation_field.GetSpacing(),   sitk_fixed.GetSpacing(),   "spacing"),
                (deformation_field.GetOrigin(),    sitk_fixed.GetOrigin(),    "origin"),
                (deformation_field.GetDirection(), sitk_fixed.GetDirection(), "direction"),
            ]:
                if a != b:
                    print(f"Warning: deformation field {name} differs from fixed image {name}: {a} vs {b}")

            displacement_transform = sitk.DisplacementFieldTransform(deformation_field)
            sitk.WriteTransform(displacement_transform, str(deformation_field_path))

            import os, tempfile

            flag = Path(str(output[0]))
            flag.parent.mkdir(parents=True, exist_ok=True)

            # Write to a temp file, fsync, then atomic rename -> avoids visibility races on NFS
            with tempfile.NamedTemporaryFile(dir=str(flag.parent), delete=False) as tf:
                tf.write(b"OK\n")
                tf.flush()
                os.fsync(tf.fileno())
                tmp_name = tf.name

            # fsync the directory entry, then rename atomically
            dir_fd = os.open(str(flag.parent), os.O_RDONLY)
            try:
                os.replace(tmp_name, str(flag))
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)




import ants

rule inverse_transform_mesh_rule:
    input: PROCESS_DIR / "{subject}/3_nonrig.flag"
    output: (PROCESS_DIR / "{subject}/4_inv_mesh.flag")
    params:
        in_ct=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/nl_transformation_params.txt",
        out_head=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/head_affine_registered.obj",

    threads: 3
    run:
        Path(str(params.out_head)).unlink(missing_ok=True)
        subject_dir = Path(input[0]).parent

        or_ref_mesh = REF_IMAGE / "oriented_mesh.obj"
        deformation_field = subject_dir / "deformationField.h5"
        elastix_field = subject_dir / "elastix_deformation_field.h5"

        transforms = [str(deformation_field)]
        if elastix_field.exists():
            transforms.append(str(elastix_field))

        slicer_cmd = [
            "xvfb-run","-a", 
            str(SLICER_CORE),
            "--no-main-window",
            "--no-splash",
            "--python-script",
            str(MESH_TRANSFORM_SCRIPT),
            str(or_ref_mesh),
            str(params.out_head),
        ] + transforms

        subprocess.run(slicer_cmd, check=True)
        if Path(params.out_head).exists() is False:
            raise ValueError("Mesh transformation failed!")

        deformation_field.unlink(missing_ok=True)
        elastix_field.unlink(missing_ok=True)
        Path(output[0]).touch(exist_ok=True)
