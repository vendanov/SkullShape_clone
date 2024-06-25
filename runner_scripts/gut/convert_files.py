import os
import glob
import pathlib

from tifffile import imread
import nibabel as nib
import numpy as np
from pathlib import Path
def tif_to_nifti(tif_file, output_dir):
    # Read the .tif file
    img = imread(tif_file)

    # Convert to NIfTI format
    nifti_img = nib.Nifti1Image(img, affine=np.eye(4))
    out_name = Path(tif_file).name.split('.')[0] + ".nii.gz"
    # Save the NIfTI image
    nib.save(nifti_img, os.path.join(output_dir, out_name))

def process_directory(directory):
    # Get all .tif files in the directory
    tif_files = glob.glob(os.path.join(directory, '*.tif'))

    for tif_file in tif_files:
        print(tif_file)
        # Create a new directory for each .tif file
        output_dir = os.path.join(directory, os.path.splitext(os.path.basename(tif_file))[0])
        os.makedirs(output_dir, exist_ok=True)

        # Convert the .tif file to NIfTI format and save it in the new directory
        tif_to_nifti(tif_file, output_dir)


# Replace 'my_directory' with the path to your directory
process_directory('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/Data embryonic gut from USA')
