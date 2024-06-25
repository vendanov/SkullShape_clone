#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pathlib import Path
import pandas as pd
import nibabel as nib
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from skimage import exposure, filters, morphology
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# In[3]:


base_folder = Path(r"/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp")
skull_file = base_folder / "combined_skull_analysis.xlsx"
skull_df = pd.read_excel(skull_file)


# # Filter subjects with skull data which have sutures

# In[4]:


skull_df["sutures_status"] = skull_df['Comment'].apply(lambda x: ("sutures"in str(x).lower() and "open" in str(x).lower()))
skull_df = skull_df[skull_df['proc_dist']<3000]


# In[5]:


# get 2 classes for sutures (equal number of sutures and no sutures)
sutures_df = skull_df[skull_df['sutures_status']==True]
no_sutures_df = skull_df[skull_df['sutures_status']==False]
no_sutures_df = no_sutures_df.sample(n=sutures_df.shape[0], random_state=42)


# In[6]:


# generate train and test split
train_df = pd.concat([sutures_df.iloc[:int(sutures_df.shape[0]*0.8)], no_sutures_df.iloc[:int(no_sutures_df.shape[0]*0.8)]])
test_df = pd.concat([sutures_df.iloc[int(sutures_df.shape[0]*0.8):], no_sutures_df.iloc[int(no_sutures_df.shape[0]*0.8):]])
# randomize the dataset
train_df = train_df.sample(frac=1, random_state=42)
test_df = test_df.sample(frac=1, random_state=42)


# In[7]:


# resize the images to 128x128x128
def resize_image(img, size=(128, 128, 128)):
    """
    Resize a 3D image to the given size.

    Parameters:
    img (numpy.ndarray): The input 3D image.
    size (tuple): The target size for each axis (depth, height, width).

    Returns:
    numpy.ndarray: The resized 3D image.
    """
    img = torch.tensor(img, dtype=torch.float32)  # Convert to a tensor
    img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    img = F.interpolate(img, size=size, mode='trilinear', align_corners=False)  # Resize
    img = img.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    return img.numpy()


def resize_data(data, size=128):
    for i in range(len(data)):
        data[i]['image'] = resize_image(data[i]['image'], size)
    return data

def read_nifty_image(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def find_location_subject(cohort_id,subject,folder_par: Path):
    ar = list(folder_par.glob(f"*{cohort_id}*/skull_shape_workdir/{subject}"))
    subj = None
    if len(ar)>0:
        subj = ar[0]
    return subj

def find_3d_bounding_box(mask):
    # Get the indices of the non-zero elements
    non_zero_indices = np.argwhere(mask)
    
    # Get the minimum and maximum indices along each axis
    z_min, y_min, x_min = non_zero_indices.min(axis=0)
    z_max, y_max, x_max = non_zero_indices.max(axis=0)
    
    # Return the bounding box coordinates
    return (z_min, y_min, x_min), (z_max, y_max, x_max)


def read_nifty_subject(folder: Path):
    
    mask = read_nifty_image(folder / "mask_skull_top.nii.gz")
    img = read_nifty_image(folder / "coregistered_head.nii.gz")
    
    return img ,mask
    


# In[ ]:





# In[8]:


# In[ ]:
size = (96, 48, 128)

def read_subjs(df):
    results = []
    for i in tqdm(range(df.shape[0])):
        subj = dict(df.iloc[i])
        s_id = subj['Unnamed: 0']
        coh = subj['COHORT_ID']
        clas_res = subj['sutures_status']
        
        loc_dir = find_location_subject(coh,s_id,base_folder)
        if loc_dir:
            img,mask = read_nifty_subject(loc_dir)
            for _ in range(4):
                mask = morphology.binary_dilation(mask)
            ids = find_3d_bounding_box(mask)
            print(np.array(ids[1]) - np.array(ids[0]))
            img = img[ids[0][0] : ids[1][0] , ids[0][1] :ids[1][1] , ids[0][2] :ids[1][2] ]
            img[img<34]=34
            img[img>50] = 50
            #img = img[ids[0][0]-3: ids[1][0]+ 3,ids[0][1]- 3:ids[1][1]+ 3,ids[0][2]- 3:ids[1][2]+ 3]
            img = resize_image(img, size=size)
            res = { "class": clas_res,  "image": img}
            results.append(res)
    return results
train_data = read_subjs(train_df)
with open(r'/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/train_data_sutures.pkl', 'wb') as f:
    pickle.dump(train_data, f)
test_data = read_subjs(test_df)
# pickle the data


with open(r'/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/test_data_sutures.pkl', 'wb') as f:
    pickle.dump(test_data, f)
    


# In[ ]:


# load the data
with open(r'/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/train_data_sutures.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open(r'/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/test_data_sutures.pkl', 'rb') as f:
    test_data = pickle.load(f)


# In[ ]:


train_data = resize_data(train_data)
test_data = resize_data(test_data)


# In[52]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num = 124
        self.shape = 32*num*num*num
        self.conv1 = nn.Conv3d(1, 16, 3)  # Use stride to reduce dimensions instead of pooling
        self.dropout1 = nn.Dropout3d(0.15)
        self.conv2 = nn.Conv3d(16, 32, 3)  # Use stride to reduce dimensions instead of pooling
        self.fc1 = nn.Linear(self.shape, 128)  # The size here depends on the output of the last conv layer
        self.fc2 = nn.Linear(128, 2)  # Output layer for binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        print(x.shape)
        x = x.view(-1, self.shape)  # Flatten the tensor
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        return x


# In[53]:


class MouseSuturesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        image = self.data[idx]['image']
        min_im = image.min()
        max_im = image.max()
        image = (image - min_im)/(max_im - min_im)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)  # Add channel dimension
        label = self.data[idx]['class']
        label = self._convert_class_to_int(label)
        return image, label

    def _convert_class_to_int(self, class_value):
        return torch.tensor(int(class_value), dtype=torch.long)
train_dataset = MouseSuturesDataset(train_data)
test_dataset = MouseSuturesDataset(test_data)


# In[54]:


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)


# In[ ]:


# Define your model
net = Net().to(device)
# Define your loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(100):
    train_loss = 0.0
    test_loss = 0.0
    net.train()  # Set model to training mode
    for i, data in enumerate(tqdm(train_loader), 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device),data[1].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update train loss
        train_loss += loss.item()

    # Calculate average train loss
    train_loss /= len(train_loader)

    # Evaluate on test set
    net.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device),data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # Calculate average test loss
    test_loss /= len(test_loader)

    # Print train and test losses
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}")

print('Finished Training')


# In[ ]:


for i in range(len(test_dataset)):
    if test_dataset[i][1] == 0:
        a = net(test_dataset[i][0])
        # print index of the class with the highest probability
        
        print(a.argmax(dim=1))


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:




