from configparser import ConfigParser
from pathlib import Path
from typing import Dict
import numpy as np
import time
import SimpleITK as sitk
import torch
from PIL import Image
from typing import Union

from sympy.physics.vector.printing import params
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
import torchvision.transforms as T

PROCESS_DIR = Path(config['cohort_dir'])

REF_IMAGE = Path(config['ref_image_dir'])
PARAMETER_FILE_NONLIN = Path(config['parameter_file_nonlin'])
PARAMETER_FILE_LIN = Path(config['parameter_file_lin'])
MESH_TRANSFORM_SCRIPT = config['mesh_transform_script']
SLICER_CORE = config['slicer_core']

# Reference image paths
ref_im = REF_IMAGE / config['ref_im']
ref_mask = REF_IMAGE / config['ref_mask']
ref_landmarks_cfg = config.get('ref_landmarks')
ref_landmarks = REF_IMAGE / ref_landmarks_cfg if ref_landmarks_cfg else None
# Additional file paths


class ImageProperties:
    def __init__(self, ini_file):
        config = ConfigParser()
        config.read(ini_file,encoding='cp1252')

        # System
        self.scanner = config.get('System', 'Scanner')
        self.instrument_sn = config.get('System', 'Instrument S/N')
        self.home_directory = config.get('System', 'Home Directory')
        self.source_type = config.get('System', 'Source Type')
        self.camera_pixel_size = float(config.get('System', 'Camera Pixel Size (um)'))

        # Acquisition
        self.data_directory = config.get('Acquisition', 'Data Directory')
        self.filename_prefix = config.get('Acquisition', 'Filename Prefix')
        self.number_of_files = int(config.get('Acquisition', 'Number Of Files'))
        self.number_of_rows = int(config.get('Acquisition', 'Number Of Rows'))
        self.number_of_columns = int(config.get('Acquisition', 'Number Of Columns'))
        self.filename_index_length = int(config.get('Acquisition', 'Filename Index Length'))
        #self.frame_index_length = int(config.get('Acquisition', 'Frame Index Length'))
        #self.partial_width = config.get('Acquisition', 'Partial Width')
        self.image_crop_origin_x = int(config.get('Acquisition', 'Image crop origin X'))
        self.image_crop_origin_y = int(config.get('Acquisition', 'Image crop origin Y'))
        self.camera_binning = config.get('Acquisition', 'Camera binning')
        self.image_rotation = float(config.get('Acquisition', 'Image Rotation'))
        self.optical_axis = int(config.get('Acquisition', 'Optical Axis (line)'))
        self.object_to_source = float(config.get('Acquisition', 'Object to Source (mm)'))
        self.camera_to_source = float(config.get('Acquisition', 'Camera to Source (mm)'))
        self.source_voltage = int(config.get('Acquisition', 'Source Voltage (kV)'))
        self.source_current = int(config.get('Acquisition', 'Source Current (uA)'))
        self.image_pixel_size = float(config.get('Acquisition', 'Image Pixel Size (um)'))
        #self.scaled_image_pixel_size = float(config.get('Acquisition', 'Scaled Image Pixel Size (um)'))
        self.image_format = config.get('Acquisition', 'Image Format')
        self.depth = int(config.get('Acquisition', 'Depth (bits)'))
        self.exposure = int(config.get('Acquisition', 'Exposure (ms)'))
        self.rotation_step = float(config.get('Acquisition', 'Rotation Step (deg)'))
        self.use_360_rotation = config.get('Acquisition', 'Use 360 Rotation')
        #self.scanning_position = float(config.get('Acquisition', 'Scanning position'))
        #self.frame_averaging = config.get('Acquisition', 'Frame Averaging')
        #self.flat_field_correction = config.get('Acquisition', 'Flat Field Correction')
        #self.geometrical_correction = config.get('Acquisition', 'Geometrical Correction')
        self.filter = config.get('Acquisition', 'Filter')


def convert_dataset_to_image(dataset: Dict[int,np.ndarray],image_props:ImageProperties):

    #check missing
    arr  = sorted(dataset)
    for i in range(len(sorted(arr))-1):
        if arr[i+1] - arr[i] > 1:
            print("Missing values between", arr[i], "and", arr[i+1])
            raise Exception()

    res_image = []
    for id in sorted(dataset):
        res_image.append( dataset[id] )

    res_array = np.array(res_image)
    affine_m = np.eye(4)
    affine_m[0,0] = affine_m[1,1] = affine_m[2,2] = image_props.image_pixel_size

    res_im = sitk.GetImageFromArray(res_array)
    voxel_size = (image_props.image_pixel_size/1000, image_props.image_pixel_size/1000,
                  image_props.image_pixel_size/1000)
    #res_im.SetDirection(affine_m.flatten())
    res_im.SetSpacing(voxel_size)
    return res_im


def predict_image(model, image, device, threshold, num_classes):
    """
    Predicts bounding boxes and labels for an image using a trained model.

    Args:
        model: Trained object detection model.
        image: PIL Image to be predicted.
        device: Device to perform inference (e.g., "cpu" or "cuda").
        threshold: Confidence threshold for filtering predictions.
        num_classes: Total number of object classes.

    Returns:
        Tuple of filtered bounding boxes and labels.
    """
    # Convert image to tensor
    image_tensor = T.ToTensor()(image).unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)

    # Get boxes, scores, and labels
    boxes = prediction[0]["boxes"]
    scores = prediction[0]["scores"]
    labels = prediction[0]["labels"]

    # Filter by confidence threshold
    valid_indices = scores > threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]

    # Enforce exactly one detection per class
    final_boxes = []
    final_labels = []

    for cls in range(1, num_classes + 1):  # Skip background (class 0)
        cls_indices = labels == cls
        if cls_indices.sum() > 0:
            cls_boxes = boxes[cls_indices]
            cls_scores = scores[cls_indices]
            # Select the box with the highest confidence
            best_idx = torch.argmax(cls_scores)
            final_boxes.append(cls_boxes[best_idx].tolist())
            final_labels.append(cls)

    return final_boxes, final_labels


def convert_result_to_pascal_voc(boxes, labels, class_names, image_size, output_path, image_name,folder_name):
    """
    Converts predicted boxes and labels to Pascal VOC XML format.

    Args:
        boxes (list or numpy.ndarray): List of bounding boxes in the format [xmin, ymin, xmax, ymax].
        labels (list or numpy.ndarray): List of corresponding class labels for the boxes.
        class_names (list): List of class names where index corresponds to the label.
        image_size (tuple): Image size as (width, height, depth).
        output_path (str): Path to save the Pascal VOC XML file.
        image_name (str): Name of the image (without the extension).

    Returns:
        None
    """
    # Create the root XML element
    annotation = ET.Element("annotation")

    # Add image metadata
    folder = ET.SubElement(annotation, "folder")
    folder.text = folder_name

    filename = ET.SubElement(annotation, "filename")
    filename.text = f"{image_name}"

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image_size[0])
    height = ET.SubElement(size, "height")
    height.text = str(image_size[1])
    depth = ET.SubElement(size, "depth")
    depth.text = "1"

    # Add each object
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(annotation, "object")

        name = ET.SubElement(obj, "name")
        name.text = class_names[label]

        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(box[0]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(box[1]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(box[2]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(box[3]))

    # Save the XML to a file
    tree = ET.ElementTree(annotation)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

def convert_head_folder(fold : Union[str,Path]):
    """

    :param fold: folder with a head
    :return: Image
    """
    i_f = fold
    if isinstance(fold,str):
        i_f = Path(i_f)
    #folder already preprocessed
    rec_folder = [ x for x in i_f.iterdir() if x.is_dir() and x.name.endswith("Rec")][0]


    parent_folder = rec_folder

    config_files = [x for x in rec_folder.iterdir() if x.name.endswith(".log")]
    if len(config_files) == 0: # for manual files
        config_files = [x for x in rec_folder.iterdir() if x.name.endswith(".log")]
    config_file = config_files[0]
    config = ImageProperties(config_file)

    #image files
    dataset = {}
    for image_file in parent_folder.glob("*.bmp"):
        #extra filter of head folder
        if image_file.name.endswith("spr.bmp"):
            continue
        ids = (int(image_file.name.replace(".bmp",'').split('rec')[-1]))
        arr = np.array(Image.open(image_file))
        dataset[ids] = arr

    return convert_dataset_to_image(dataset,config),config


def process_cohort_folder():
    results = []
    not_cohort = ["BMD","TMD","Morphology","csv","+","workdir","checkpoints","_Screen","histogram"]
    for fold in PROCESS_DIR.iterdir():
        if not fold.is_dir():
            continue
        if any(substring in fold.name for substring in not_cohort):
            continue
        results.append(fold)
    return results

def flip_sitk_image( im: sitk.Image, axis):
    """
    Flip an image along specified axes and preserve spatial alignment.

    Parameters:
        im (sitk.Image): Input image.
        axis (Tuple[bool, bool, bool]): Axes along which to flip (e.g., (True, False, False)).

    Returns:
        sitk.Image: Flipped image.
    """
    # Flip the image
    flipped_image = sitk.Flip(im,axis)

    # Ensure spatial metadata (origin, spacing, direction) is consistent
    flipped_image.SetOrigin(im.GetOrigin())
    flipped_image.SetSpacing(im.GetSpacing())
    flipped_image.SetDirection(im.GetDirection())

    return flipped_image

def flip_image(im_file_in: str, im_file_out: str, axis):
    """
    Flip an image along specified axes and preserve spatial alignment.

    Parameters:
        im_file_in (str): Path to the input image.
        im_file_out (str): Path to save the flipped image.
        axis (Tuple[bool, bool, bool]): Axes along which to flip (e.g., (True, False, False)).
    """
    if Path(im_file_out).exists():
        Path(im_file_out).unlink()

    # Read the input image
    image = sitk.ReadImage(im_file_in)

    # Flip the image
    flipped_image = flip_sitk_image(image, axis)

    # Save the flipped image
    sitk.WriteImage(flipped_image,im_file_out)


SRC_SUBJECTS = process_cohort_folder()
SRC_SUBJECTS = [fold.name for fold in SRC_SUBJECTS if fold.is_dir() and (len(list(fold.iterdir())) > 0)]

SRC_SUBJECTS = [x for x in SRC_SUBJECTS if ('_' in x)]
SRC_SUBJECTS = [x for x in SRC_SUBJECTS if ('BMD' not in x) and ('!' not in x)]
SRC_SUBJECTS = [x for x in SRC_SUBJECTS if x.startswith('_') is False]

#make_output folder

OUT_FOLD = PROCESS_DIR / "full_body_workdir"
OUT_FOLD.mkdir(exist_ok=True)

def generate_projection_image(input_image: sitk.Image, axis=1):
    # find the first image
    projection_filter = sitk.MaximumProjectionImageFilter()

    projection_filter.SetProjectionDimension(axis)
    projection_image = projection_filter.Execute(input_image)
    return projection_image




rule all:
    input:
        expand(OUT_FOLD/"{subject}/detect_head_and_paws.flag",subject = SRC_SUBJECTS)

rule make_nifti_image:
    input:
        PROCESS_DIR/"{subject}"
    output:
        FLAG = (OUT_FOLD/"{subject}/make_nifti_image.flag")
    run:
        start_ts = time.time()
        print(f"[INFO] make_nifti_image start subject={wildcards.subject}")

        Path(output.FLAG).parent.mkdir(exist_ok=True)
        Path(output.FLAG).touch(exist_ok=True)

        img, config_= convert_head_folder(input[0])
        if '1176' in config_.scanner:
            print("Flipping image")
            img = flip_sitk_image(img,(True,False,True))

        sitk.WriteImage(img,Path(output.FLAG).parent / "full_body_image_raw.nii.gz")  # real output
        print(f"[INFO] make_nifti_image done subject={wildcards.subject} elapsed={time.time() - start_ts:.1f}s")




rule detect_head_and_paws:
    input:
        OUT_FOLD/"{subject}/make_nifti_image.flag"
    output:
        FLAG = (OUT_FOLD/"{subject}/detect_head_and_paws.flag")
    params:
        result_xml = lambda wildcards: OUT_FOLD / f"{wildcards.subject}/full_body_top_classes.xml"
    run:
        start_ts = time.time()
        print(f"[INFO] detect_head_and_paws start subject={wildcards.subject}")
        #run the script
        fold_ = Path(input[0]).parent
        full_body_image = sitk.ReadImage(fold_/"full_body_image_raw.nii.gz")
        print(str(params.result_xml))
        # Load Faster R-CNN with ResNet-50 backbone
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        # Adjust the model for your number of classes
        num_classes = 6  # Number of classes, including the background
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

        # Load the model weights
        model.load_state_dict(torch.load(config['body_part_detection_top_model']))

        # Set the model to evaluation
        model.eval()
        projection_image = generate_projection_image(full_body_image, axis=1)
        image = Image.fromarray(sitk.GetArrayFromImage(projection_image)[:, 0, :])

        boxes, labels = predict_image(model,image,'cpu',threshold=0.1,num_classes=num_classes)
        if len(boxes) < num_classes - 1:
            print(f"Warning {input[0]} due to insufficient number of labels")
            print(boxes,labels)

        classes = ['__background__', '1_hand', '1_leg', '2_hand', '2_leg', 'head']
        convert_result_to_pascal_voc(boxes,labels,classes,
            image.size,str(params.result_xml) ,'image_file','.')


        Path(output.FLAG).touch(exist_ok=True)
        print(f"[INFO] detect_head_and_paws done subject={wildcards.subject} elapsed={time.time() - start_ts:.1f}s")

