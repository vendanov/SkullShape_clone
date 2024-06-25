import sys
from pathlib import Path
from typing import List


#PROCESS_DIR = Path(config['cohort_dir'])

##global vars
REF_IMAGE = Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/reference_image/skull_top_label.nii.gz')

FOLDER = Path('/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/')

# generate list of all the in FORLDER' subfolders which contain the folder 'skull_shape_workdir' and the folder
# 'skull_shape_workdir' get skull_shape_workdir subfolders which containsthe file 'nl_transformation_params.txt' (use only Pathlib)

def get_folders_with_nl_transformation_params(FOLDER: Path) -> List[Path]:
    folders = []
    for folder in FOLDER.iterdir():
        if folder.is_dir():
            if (folder / 'skull_shape_workdir').exists():
                for subfolder in (folder / 'skull_shape_workdir').iterdir():
                    if subfolder.is_dir():
                        if (subfolder / 'nl_transformation_params.txt').exists():
                            folders.append(subfolder)


    return folders

folders_with_subs = get_folders_with_nl_transformation_params(FOLDER)

# generate snakemake rule for each folder in folders_with_subs





rule all:
    input:
        expand("{dirname}/mask_skull_top.nii.gz",dirname=folders_with_subs)


rule generate_mask_skull_top:
    input:
        "{dirname}/nl_transformation_params.txt"
    output:
        "{dirname}/mask_skull_top.nii.gz"
    run:
        # read the nl_transformation_params.txt file
        text = Path(input[0]).read_text()

        # replace parameters in the text (FinalBSplineInterpolationOrder 3) to (FinalBSplineInterpolationOrder 1)
        text = text.replace('FinalBSplineInterpolationOrder 3', 'FinalBSplineInterpolationOrder 1')

        # write the text to the file
        (Path(input[0]).parent /"nl_params_nn.txt").write_text(text)

        # run the command

        cmd = "transformix -out '{dirname}' -tp '{dirname}/nl_params_nn.txt' -in {refim}".format(dirname=Path(input[0]).parent,refim=str(REF_IMAGE))
        shell(cmd)
        cmd = "mv '{dirname}/result.nii.gz' '{dirname}/mask_skull_top.nii.gz'".format(dirname=Path(input[0]).parent)
        shell(cmd)
        (Path(input[0]).parent /"nl_params_nn.txt").unlink()





