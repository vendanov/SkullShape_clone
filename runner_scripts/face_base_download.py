import numpy as np
import requests

import pandas as pd
import argparse
from pathlib import Path
import nibabel as nib


parser = argparse.ArgumentParser(description='Download face base data')
parser.add_argument("face_base_csv", type=str, help="Path to the face base csv file")
parser.add_argument("output_dir", type=str, help="Path to the output directory")

args = parser.parse_args()

face_base_csv = args.face_base_csv
output_dir = Path(args.output_dir)



df = pd.read_csv(face_base_csv)

# Function to download a single file
def download_file(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")


# Base URL for the face base data
base_url = "https://www.facebase.org/"



# iterate over rows of the dataframe

for index, row in df.iterrows():
    # Construct the full URL

    url = f"{base_url}{row['url']}"

    dataset = row['dataset']
    dataset = dataset.replace("-", "_")



    ds_dir = output_dir / dataset
    ds_dir.mkdir(exist_ok=True)

    replicate = row['replicate']
    replicate = replicate.replace("-", "_")

    replicate_dir = ds_dir / replicate
    replicate_dir.mkdir(exist_ok=True)
    mnc_file = replicate_dir / "image.mnc"
    if (replicate_dir / "image.mnc").exists():
        print(f"Skipping {url} as it already exists")
    else:
        download_file(url, str(mnc_file))

    (ds_dir / 'skull_shape_workdir'/replicate).mkdir(exist_ok=True, parents=True)
    if (ds_dir / 'skull_shape_workdir'/replicate/'make_nifti_image.flag').exists():
        print(f"Skipping {str(ds_dir / 'skull_shape_workdir'/replicate/'make_nifti_image.flag')} as it already exists")
        continue

    print("Loading MNC file")

    # Load the MNC file
    mnc_image = nib.load(mnc_file)
    # Get the image data as a NumPy array
    img_data = mnc_image.get_fdata()

    # Normalise data to fit uint8 range [0, 255]
    data_min = img_data.min()
    data_max = img_data.max()
    if data_max > data_min:  # Avoid division by zero
        data_scaled = (img_data - data_min) / (data_max - data_min) * 255

    else:
        data_scaled = np.zeros_like(img_data)  # If all values are the same

    threshold = 30
    data_scaled[data_scaled <  threshold] = threshold
    data_scaled = data_scaled -threshold
    data_uint8 = data_scaled.astype(np.uint8)

    # Create a new NIFTI image
    new_nifti_img = nib.Nifti1Image(data_uint8, mnc_image.affine,mnc_image.header)

    # convert to uint 8
    print("Loaded MNC file")
    # Create a new NIfTI image from the modified data
    mnc_image = new_nifti_img

    # Save as NIfTI
    work_dir = Path.cwd()
    if (ds_dir / 'skull_shape_workdir'/replicate/'make_nifti_image.flag').exists():
        print(f"Skipping {str(ds_dir / 'skull_shape_workdir'/replicate/'make_nifti_image.flag')} as it already exists")
        continue
    nifti_file = (ds_dir / 'skull_shape_workdir'/replicate) / "ct_image.nii.gz"
    print(f"Saving NIfTI file to {nifti_file}")
    nib.save(mnc_image, nifti_file)
    print(f"Saved NIfTI file to {nifti_file}")
    # touch empty file
    with open((ds_dir / 'skull_shape_workdir'/replicate/'make_nifti_image.flag'), 'w') as f:
        pass




    # convert mnc to nifti

    print(f"Downloaded {url}")




print("Download complete")

