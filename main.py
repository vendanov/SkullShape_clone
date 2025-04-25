
import argparse
import os
import shutil
import subprocess
import sys
import yaml

import tempfile

from pathlib import Path

from mouse_skull_pipelines.scripts.gpa import gpa_new


def registration_to_reference(cohort_folder: Path, reference_mouse):

    """
    Check if the registration is done for the given cohort folder.
    """

    # read config file
    # check if the registration is done

    # current file path
    current_file = Path(__file__).resolve()
    registration_cfg = current_file.parent / 'smk_configs' / 'mouse_skull.yaml'

    script_home = current_file / 'snakemake_workflows' / 'generic_coregistration.smk'

    # read yaml file

    with open(registration_cfg, 'r') as file:
        config = yaml.safe_load(file)
        config['ref_image_dir'] = str(reference_mouse)
        config['cohort_dir'] = str(cohort_folder)

        # save the temporary config file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as temp_file:
            temp_file.write(yaml.dump(config).encode())
            temp_file_path = temp_file.name
            print(f"Temporary config file created at: {temp_file_path}")
            # run the temporary config file registration
            # run the registration
            snakemake_command = f"snakemake --snakefile {script_home} --configfile {temp_file_path} --cores 10"

            cmd = snakemake_command.split()
            subprocess.run(cmd, check=True)

            # remove the temporary config file
            os.remove(temp_file_path)



def compare_cohorts(cohort_1: str, cohort_2: str, reference_mouse: str, result_dir: str):
    """
    Compare two cohorts of mice and save the results to a file.
    """
    cohort_1 = Path(cohort_1)
    cohort_2 = Path(cohort_2)

    reference_mouse = Path(reference_mouse)
    result_dir = Path(result_dir)

    print(f"Comparing cohorts {cohort_1} and {cohort_2} with reference mouse {reference_mouse}")
    # Load the cohorts and reference mouse data

    registration_to_reference(cohort_1, reference_mouse)
    registration_to_reference(cohort_2, reference_mouse)
    # Load the cohorts and reference mouse data

    cohort_1_meshes = { x.name : x/"head_affine_registered.obj" for x in Path(cohort_1).iterdir()  if (x/"head_affine_registered.obj").exists() }
    cohort_2_meshes = { x.name : x/"head_affine_registered.obj" for x in Path(cohort_2).iterdir()  if (x/"head_affine_registered.obj").exists() }

    coh_aligned = gpa_new({**cohort_1_meshes, **cohort_2_meshes})

    # Save the aligned meshes to the result directory
    # compute the statistics and compare the distributions










def main():
    parser = argparse.ArgumentParser(description="Run the full body and skull pipeline.")
    sbp = parser.add_subparsers("subcommand", required=True)

    cmp_cohorts = sbp.add_parser("compare_cohorts", help="Run the pipeline.")
    cmp_cohorts.add_argument(
        "--cohort_1",
        type=str,
        required=True,
        help="Path to the cohort directory.",
    )
    cmp_cohorts.add_argument(
        "--cohort_2",
        type=str,
        required=True,
        help="Path to the cohort directory.",
    )
    cmp_cohorts.add_argument(
        "--reference_mouse",
        type=str,
        required=True,
        help="Path to the reference mouse directory.",
    )


    cmp_cohorts.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to the result file.",
    )
    args = parser.parse_args()
    print(args)
    # Check if the result directory exists, if not create it
    if args.cmd == "compare_cohorts":
        result_dir = Path(args.result_dir)
        if not result_dir.exists():
            os.makedirs(result_dir)
        # Call the compare_cohorts function
        compare_cohorts(args.cohort_1, args.cohort_2, args.reference_mouse, result_dir)









if __name__ == "__main__":
    main()




