from pathlib import Path

import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator

import torchvision.transforms as T
from torchvision.models import vgg16
import nibabel as nib
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .image_algorithms import marching_cubes, generate_projection_image, read_sitk_image, convert_2d_to_PIL, \
    compute_2d_transform_no_opencv, convert_2d_to_3d_transform_no_opencv
from .io import read_classes_from_xmls, convert_result_to_pascal_voc, read_xml_class
from .mesh_algorithms import sample_points_from_mesh, normalize_point_cloud


class TNet(nn.Module):
    """
    Input Transformation Network as in PointNet.
    Predicts a transformation matrix to align the input point cloud.
    """
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, k * k)

        # Initialize the weights/bias with identity transformation
        nn.init.constant_(self.fc3.weight, 0)
        with torch.no_grad():
            self.fc3.bias.copy_(torch.eye(k).flatten())

    def forward(self, x):
        # x: [B, k, N]
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))      # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))      # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))      # [B, 1024, N]
        x = torch.max(x, 2)[0]                    # [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))        # [B, 512]
        x = F.relu(self.bn5(self.fc2(x)))        # [B, 256]
        x = self.fc3(x)                           # [B, k*k]

        x = x.view(-1, self.k, self.k)           # [B, k, k]
        return x

class PointNetRegressor(nn.Module):
    """
    PointNet model for regression tasks.
    """
    def __init__(self, num_points=1024, num_outputs=3, input_dim=3):
        super(PointNetRegressor, self).__init__()
        self.num_points = num_points
        self.num_outputs = num_outputs

        # Input Transformation
        self.input_transform = TNet(k=input_dim)

        # Shared MLP layers
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # Global Feature
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, self.num_outputs)

    def forward(self, x):
        """
        Forward pass of PointNetRegressor.

        Parameters:
        x: [B, N, 3] tensor representing a batch of point clouds

        Returns:
        out: [B, num_outputs] tensor representing regression outputs
        """
        # Transpose to [B, 3, N]
        x = x.transpose(2, 1)
        #print(f"After transpose: {x.shape}")  # Should be [B, 3, N]

        # Input Transformation
        input_transform = self.input_transform(x)  # [B, 3, 3]
        #print(f"Input Transform shape: {input_transform.shape}")  # Should be [B, 3, 3]

        x = torch.bmm(input_transform, x)          # [B, 3, N]
        #print(f"After transformation: {x.shape}")  # Should be [B, 3, N]

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
        #print(f"After Conv1: {x.shape}")           # Should be [B, 64, N]

        x = F.relu(self.bn2(self.conv2(x)))        # [B, 128, N]
        #print(f"After Conv2: {x.shape}")           # Should be [B, 128, N]

        x = F.relu(self.bn3(self.conv3(x)))        # [B, 1024, N]
        #print(f"After Conv3: {x.shape}")           # Should be [B, 1024, N]

        # Global Feature
        x = torch.max(x, 2)[0]                      # [B, 1024]
        #print(f"After Max Pooling: {x.shape}")      # Should be [B, 1024]

        # Fully Connected Layers
        x = F.relu(self.bn4(self.fc1(x)))          # [B, 512]
        #print(f"After FC1: {x.shape}")             # Should be [B, 512]

        x = F.relu(self.bn5(self.fc2(x)))          # [B, 256]
        #print(f"After FC2: {x.shape}")             # Should be [B, 256]

        x = self.fc3(x)                             # [B, num_outputs]
        #print(f"After FC3: {x.shape}")             # Should be [B, num_outputs]

        return x



    def compute_from_image(self, image_path: str):
        """
        Compute the regression output from an input image.

        Parameters:
        image_path: str, path to the input image

        Returns:
        output: np.array, regression output
        """
        # Load the image
        #
        # Preprocess the image

        mesh,image_data= marching_cubes(image_path)
        print("Image data: ", image_data)
        # Forward pass
        point_cloud = sample_points_from_mesh(mesh, point_count=self.num_points)
        #print(f"point_cloud: {point_cloud}")
        points, center, scale = normalize_point_cloud(point_cloud)
        #print("Center: ", center)
        #print("Scale: ", scale)
        points = torch.from_numpy(points).float()
        # rescale points
        self.eval()

        output = points.unsqueeze(0)
        #print(f"input shape: {output.shape}")
        #print(f"input: {output}")
        output = self.forward(output)
        output = output.cpu().detach().numpy()

        # reshape to (n, 3)
        output = output.reshape(-1, 3)
        output = output * scale + center

        #output =apply_transform_to_point(output, image_data, inverse=True)

        return output

    @staticmethod
    def landmark_point_generation(input_file: str, control_points_model: str):
        net = PointNetRegressor(1000, 9)
        net.load_state_dict(torch.load(control_points_model))
        with torch.no_grad():
            net.eval()

            pts = net.compute_from_image(input_file)
            return pts

# %%
import numpy as np
import nibabel as nib
import vtk
import torch


def get_head_file_n_mesh(subj_id, cohort_id):
    """
    Get the head file and mesh file for a given subject and cohort
    :param subj_id:
    :param cohort_id:
    :return:
    """
    folders = [x for x in base_folder.iterdir() if cohort_id in x.name and (x / "skull_shape_workdir").exists()]
    for fold in folders:
        files = list((fold / "skull_shape_workdir").iterdir())
        for file in files:
            if str(subj_id) in file.name:
                if (file / "head_affine_registered.obj").exists() and (file / "coregistered_head.nii.gz").exists():
                    return file / "head_affine_registered.obj", file / "coregistered_head.nii.gz"
    return None, None


def get_points_from_mesh(mesh_file):
    pt1_id = [9567, 9449]
    pt2_id = [7752, 24168]
    pt3_id = [17278, 16945]

    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(mesh_file))
    reader.Update()
    polydata = reader.GetOutput()

    points = polydata.GetPoints()
    pts1 = [points.GetPoint(pt1_id[0]), points.GetPoint(pt1_id[1])]
    pts2 = [points.GetPoint(pt2_id[0]), points.GetPoint(pt2_id[1])]
    pts3 = [points.GetPoint(pt3_id[0]), points.GetPoint(pt3_id[1])]
    return np.mean(pts1, axis=0), np.mean(pts2, axis=0), np.mean(pts3, axis=0)


def world_to_voxel(world_coords, affine):
    """Convert world coordinates to voxel coordinates."""
    inv_affine = np.linalg.inv(affine)
    voxel_coords = nib.affines.apply_affine(inv_affine, world_coords)
    return voxel_coords


def plane_projection(head_data, points_world, head_affine):
    v1 = points_world[1] - points_world[0]
    v2 = points_world[2] - points_world[0]
    normal = np.cross(v1, v2)
    normal_vector = normal / np.linalg.norm(normal)

    # Calculate the plane constant
    d = -np.dot(normal, points_world[0])

    # Generate a grid of points on the plane
    num_points = 800  # Higher resolution for the slice
    grid_u, grid_v = np.meshgrid(np.linspace(-1, 2, num_points), np.linspace(-1, 1, num_points))
    grid_points = points_world[0] + np.outer(grid_u.ravel(), v1) + np.outer(grid_v.ravel(), v2)

    # Interpolate the image data at these points
    dims = head_data.shape
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    z = np.arange(dims[2])
    interpolator = RegularGridInterpolator((x, y, z), head_data, method='linear', bounds_error=False, fill_value=0)

    # Convert the grid points back to voxel coordinates
    grid_points_voxel = nib.affines.apply_affine(np.linalg.inv(head_affine), grid_points)

    # Ensure grid points are within the volume bounds
    grid_points_voxel = np.clip(grid_points_voxel, 0, np.array(dims) - 1)

    # Interpolate the values at the grid points
    slice_data = interpolator(grid_points_voxel).reshape(num_points, num_points)

    # compute d as 10mm / 100
    d = 10 / 100
    # produce 100 parallel planes to the original one to generate a projection image
    slices = []
    for i in range(100):
        slice_data = get_parallel_plane(head_data, grid_points, head_affine, normal_vector, d * (i + 1), num_points)
        slices.append(slice_data)

    # produce the projection image as the maximum value of the 100 parallel planes
    slice_data = np.max(slices, axis=0)
    # produce the projection image as the average value of the 100 parallel planes
    slice_data_mean = np.mean(slices, axis=0)
    # produce the projection image as the .75 perc value of the 100 parallel planes
    slice_data_75 = np.percentile(slices, 90, axis=0)



    return [slice_data, slice_data_mean, slice_data_75]


def get_parallel_plane(head_data, grid_points,
                       head_affine, normal_vector, d, num_points=400):
    # Interpolate the image data points shifted by the normal vector and d
    dims = head_data.shape
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    z = np.arange(dims[2])
    interpolator = RegularGridInterpolator((x, y, z), head_data, method='linear', bounds_error=False, fill_value=0)

    # shift the grid points by the normal vector and d
    grid_points_shifted = grid_points + d * normal_vector

    # Convert the grid points back to voxel coordinates
    grid_points_voxel = nib.affines.apply_affine(np.linalg.inv(head_affine), grid_points_shifted)

    return interpolator(grid_points_voxel).reshape(num_points, num_points)


def _read_plane(subj_id, cohort_id):
    mesh_file, head_file = get_head_file_n_mesh(subj_id, cohort_id)
    if mesh_file is None:
        return None
    head_img = nib.load(head_file)
    head_data = head_img.get_fdata()
    # filter fdata
    head_data[head_data < 34] = 0
    head_data[head_data > 70] = 70
    head_affine = head_img.affine
    points_world = get_points_from_mesh(mesh_file)
    plane = plane_projection(head_data, points_world, head_affine)
    return plane



def _get_points_from_mesh(mesh_file):
    pt1_id = [9567, 9449]
    pt2_id = [7752, 24168]
    pt3_id = [17278, 16945]

    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(mesh_file))
    reader.Update()
    polydata = reader.GetOutput()

    points = polydata.GetPoints()
    pts1 = [points.GetPoint(pt1_id[0]), points.GetPoint(pt1_id[1])]
    pts2 = [points.GetPoint(pt2_id[0]), points.GetPoint(pt2_id[1])]
    pts3 = [points.GetPoint(pt3_id[0]), points.GetPoint(pt3_id[1])]
    return np.mean(pts1, axis=0), np.mean(pts2, axis=0), np.mean(pts3, axis=0)



class SuturesNetWrapper:
    def __init__(self, model_path):
        self.model = vgg16()
        self.model.classifier[6] = nn.Linear(4096, 1)

        self.model.load_state_dict(torch.load(model_path))

    def to(self, device):
        self.model.to(device)


    def predict(self, image_path, mesh_path):

        # Load the image
        #
        # Preprocess the image
        head_img = nib.load(image_path)
        head_data = head_img.get_fdata()

        head_data[head_data < 34] = 0
        head_data[head_data > 70] = 70
        head_affine = head_img.affine
        pts_world = _get_points_from_mesh(mesh_path)

        image = plane_projection(head_data, pts_world, head_affine)

        # Normalize the image for each channel
        for i in range(len(image)):
            min_im = image[i].min()
            max_im = image[i].max()
            image[i] = (image[i] - min_im) / (max_im - min_im)

        # image = [image[0],image[0],image[0]]
        image = torch.tensor(np.array(image), dtype=torch.float32)

        #image = image.unsqueeze(0)  # Add channel dimension

        # Forward pass
        self.model.eval()

        output = self.model(image.unsqueeze(0))
        output = torch.sigmoid(output)
        output = output.cpu().detach().numpy()
        return (output > 0.5)[0][0]

def predict_image_boxes(model, image, device, threshold, num_classes):
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

def coalign_roi_based(ref_path,flo_path,ref_rois,
                      model_path, mask_path,
                      res_path, res_mask_path, out_im_name = "flo_proj.png",axis=1):

    refim = read_sitk_image(ref_path)
    floim = read_sitk_image(flo_path)

    flo_proj = generate_projection_image(floim, axis=axis)


    classes  = ['__backgr'] + read_classes_from_xmls(ref_rois)

    # Load the model
    model = fasterrcnn_resnet50_fpn(weights=None)

    # Adjust the model for your number of classes
    num_classes = len(classes)  # Number of classes, including the background
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    # compute the bounding boxes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    flo_proj_im = convert_2d_to_PIL(flo_proj)

    flo_boxes, flo_labels = predict_image_boxes(model, flo_proj_im, device, 0.1, num_classes)
    (Path(flo_path).parent/out_im_name).unlink(missing_ok=True)
    flo_proj_im.save( str(Path(flo_path).parent/out_im_name))
    convert_result_to_pascal_voc(flo_boxes, flo_labels, classes, flo_proj.shape, str(Path(flo_path).parent/"flo_proj.xml"), out_im_name,str(Path(flo_path).parent))

    # read ref_boxes and ref_labels from xml
    ref_boxes, ref_labels = read_xml_class(ref_rois,classes)

    print(flo_labels)
    if len(flo_labels)> 1:

        # convert boxes to center points

        flo_centers = [ [(box[0] + box[2])/2, (box[1] + box[3])/2] for box in flo_boxes]
        ref_centers = [ [(box[0] + box[2])/2, (box[1] + box[3])/2] for box in ref_boxes]

        #get points in world coordinates
        print("flo centers",flo_centers)

        ref_aff = nib.load(ref_path).affine
        flo_aff = nib.load(flo_path).affine
        def get_world_coords_from_image_coords(coords, affine):
            print('axis',axis)
            if axis == 1:
                proj = [affine @ np.array([int(x),0,int(y),1]) for x,y in coords]
                res = [[x[0],x[2]] for x in proj]
            elif axis == 0:



                proj = [affine @ np.array([0,int(x),int(y),1]) for x,y in coords]
                print("proj",proj)
                res = [[x[1],x[2]] for x in proj]
            elif axis == 2:
                proj = [affine @ np.array([0,int(x),int(y),1]) for x,y in coords]
                res = [[x[0],x[1]] for x in proj]
            else:
                raise ValueError("Invalid axis")
            return res


        flo_points = get_world_coords_from_image_coords(flo_centers,flo_aff)
        print("flo points",flo_points)

        ref_points = get_world_coords_from_image_coords(ref_centers,ref_aff)

        print("ref points",ref_points)

        # now make points matched to the flo points based on labels
        matched_points = []

        for label in flo_labels:
            if label in ref_labels:
                idx = ref_labels.index(label)
                matched_points.append(ref_points[idx])
            else:
                print("No matching label found")



        # compute the transformation

        transform = compute_2d_transform_no_opencv(np.array(flo_points),np.array(matched_points))
        print("transform",transform)
        # convert to 3d
        transform_3d = convert_2d_to_3d_transform_no_opencv(transform,projection_axis=axis)
        print("transform 3d",transform_3d)

        # apply the transformation nibabel
        flo_nib = nib.load(flo_path)
        flo_nib_data = flo_nib.get_fdata()
        res_aff = np.dot(transform_3d,flo_nib.affine)
        image_save = nib.Nifti1Image(flo_nib_data,res_aff)
        nib.save(image_save,res_path)

        flo_nib_mask = nib.load(mask_path)
        flo_nib_mask_data = flo_nib_mask.get_fdata()
        res_aff = np.dot(transform_3d,flo_nib_mask.affine)
        image_save = nib.Nifti1Image(flo_nib_mask_data,res_aff)
        nib.save(image_save,res_mask_path)

        return len(flo_labels) < (len(classes)-1)
    else :
        print("No boxes found in flo image")
        return None


