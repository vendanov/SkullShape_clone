import numpy as np

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

SRC_SUBJECTS = [fold.name for fold in PROCESS_DIR.iterdir() if fold.is_dir() and (fold / ct_im_name).exists()  ]


print(SRC_SUBJECTS)

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

    threads: 2
    run:
        process_folder = Path(output[0]).parent
        Path(str(params.out_head)).unlink(missing_ok=True)
        Path(str(params.out_mat)).unlink(missing_ok=True)
        Path(str(params.out_affine)).unlink(missing_ok=True)

        in_ct = Path(input[0])


        config : LinearRegistrationConfig  = LinearRegistrationConfig(is_simple=LINEAR_SIMPLE,
            ref_landmarks=None,
            side_landmarks_model=None,control_points_model=None)
        register_linear_full(str(in_ct),str(ref_im),
            str(params.in_mask),ref_mask=str(ref_mask),
            config=config,


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

rule non_rigid_registration_rule:
    input: (PROCESS_DIR / "{subject}/2_lin_cor.flag")
    output: (PROCESS_DIR / "{subject}/3_nonrig.flag")
    params:
        out_nlim=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/nonlin_image.nii.gz",
        out_txt=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/nl_transformation_params.txt",
        inp_coregim=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/coregistered_head.nii.gz"
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
    input: (PROCESS_DIR / "{subject}/3_nonrig.flag")
    output: (PROCESS_DIR / "{subject}/4_inv_mesh.flag")
    params:
        in_ct=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/nl_transformation_params.txt",
        out_head=lambda wildcards: PROCESS_DIR / f"{wildcards.subject}/head_affine_registered.obj",

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
