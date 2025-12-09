import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import nibabel as nib

# Config
# Expected keys:
#   - cohort_dir: root directory containing subject folders and 'full_body_workdir'

ROOT = Path(config["cohort_dir"]) if "cohort_dir" in config else None
if ROOT is None:
    raise ValueError("Missing 'cohort_dir' in config for extract_head_from_full_body.smk")

RESULT_ROOT = ROOT / "mouse_skull_workdir"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

FULL_BODY = ROOT / "full_body_workdir"

SUBJECTS = [
    d.name for d in FULL_BODY.iterdir()
    if d.is_dir()
    and (d / "full_body_top_classes.xml").exists()
    and (d / "full_body_image_raw.nii.gz").exists()
]

rule all:
    input:
        expand(str(RESULT_ROOT / "{subject}" / "ct_image.nii.gz"), subject=SUBJECTS)


def extract_head_bbox(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name == "head":
            bbox = obj.find("bndbox")
            xmin = round(float(bbox.find("xmin").text))
            ymin = round(float(bbox.find("ymin").text))
            xmax = round(float(bbox.find("xmax").text))
            ymax = round(float(bbox.find("ymax").text))
            return [xmin, ymin, xmax, ymax]
    return None


rule extract_head_nifti:
    input:
        xml=lambda wc: FULL_BODY / wc.subject / "full_body_top_classes.xml",
        img=lambda wc: FULL_BODY / wc.subject / "full_body_image_raw.nii.gz"
    output:
        out=str(RESULT_ROOT / "{subject}" / "ct_image.nii.gz")
    run:
        out_dir = Path(output.out).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        bbox = extract_head_bbox(str(input.xml))
        if bbox is None:
            raise ValueError(f"No 'head' label found in {input.xml}")

        img = nib.load(str(input.img))
        affine = img.affine.copy()
        for i in range(3):
            if affine[i, i] < 0:
                affine[i, i] = -affine[i, i]

        data = img.get_fdata()
        x0, y0, x1, y1 = map(int, bbox)
        # add some margin
        y_start = max(0, y0 - 10)
        y_end = min(data.shape[2], y1 + 10)
        head = data[x0:x1, :, y_start:y_end].astype(np.uint8)

        head_img = nib.Nifti1Image(head, affine)
        nib.save(head_img, str(output.out))
