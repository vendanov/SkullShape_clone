import os
import xml.etree.ElementTree as ET

def read_classes_from_xmls(annotations_file):
    """
    Reads all unique class names from Pascal VOC XML files.

    Args:
        annotations_dir (str): Path to the directory containing XML annotation files.

    Returns:
        List[str]: A list of unique class names.
    """
    classes = set()  # Use a set to avoid duplicates
    annotations_dir = [annotations_file]
    for xml_file in annotations_dir:
        if not xml_file.endswith(".xml"):
            continue

        xml_path = xml_file

        # Skip empty or invalid XML files
        if os.path.getsize(xml_path) == 0:
            print(f"Warning: Skipping empty file: {xml_file}")
            continue

        try:
            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract class names
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                classes.add(class_name)
        except ET.ParseError as e:
            print(f"Error parsing {xml_file}: {e}")
            continue

    return sorted(list(classes))  # Return as a sorted list # Return as a sorted list


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

def read_xml_class(annotation_file,classes):
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    labels = []
    boxes = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in classes:
            continue
        labels.append(classes.index(class_name))
        bbox = obj.find("bndbox")
        xmin = round(float(bbox.find("xmin").text))
        ymin = round(float(bbox.find("ymin").text))
        xmax = round(float(bbox.find("xmax").text))
        ymax = round(float(bbox.find("ymax").text))
        boxes.append([xmin, ymin, xmax, ymax])

    # boxes = torch.tensor(boxes,dtype=torch.float32)
    # labels = torch.tensor(labels,dtype=torch.int64)
    return boxes, labels
