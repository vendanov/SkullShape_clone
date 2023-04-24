import os.path
import pathlib
import re
from dataclasses import dataclass
from typing import Dict
import pickle
import SimpleITK as sitk
import pathlib

import numpy as np
import pandas as pd

cohort_list = r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/XRY normal skull WT animal.xlsx"
input_folders = "/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/"
PROCESS_DIR = pathlib.Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_workdir')

print(PROCESS_DIR)
@dataclass
class MouseDescriptor:

    cohort_id: str
    gender: str
    date: str
    subj_id: str
    subj_fold: pathlib.Path
    cohort_fold: pathlib.Path
    preprocessed: bool

class Subject:

    def __init__(self,subf):
        self.specimen = subf['SPECIMEN_ID']
        self.cohort = subf['COHORT_ID']
        self.gender = str(subf['GENDER'])
        self.date = subf['DATE']
        self.scanner = subf['Machine']


dataset : Dict[str, MouseDescriptor] = {}

def has_head(path: pathlib.Path) -> bool:
    """
    return
    :param path: path of folder
    :return: True if folder have been processed
    """

    if len(list(path.glob("**/*__head"))) > 0:
        return True
    return False


cohorts = list( pd.read_excel(cohort_list) ['COHORT_ID'])
cohort_exel = pd.read_excel(cohort_list)
import tqdm.auto
print("Starting processing files")
if os.path.exists("processed_files.pkl"):
    f = open("processed_files.pkl","rb")
    dataset = pickle.load(f)
    f.close()
else:
    for fold_ in tqdm.tqdm(list(pathlib.Path(input_folders).iterdir())):
        print(str(fold_))
        if fold_.is_file():
            continue
        fn_ =fold_.name
        #print(fn_)
        match = re.search(r'c\d+', fn_)
        if match:
            res = match.group()
            print(res)
            if res in cohorts:
                indexes = list(cohort_exel[cohort_exel['COHORT_ID'] == res].index)
                ####generate Descriptors
                folers = list(fold_.iterdir()) # folders inside cohort
                for id in indexes:
                    subj = Subject(cohort_exel.loc[id])
                    #get folder with subject
                    #print(cohort_exel.loc[id])
                    #print(fold_)
                    print("processing cohort " + subj.cohort + " and specimen " + subj.specimen )
                    subj_folds = [x for x in folers if x.name.endswith(subj.specimen)]
                    if len(subj_folds) > 0:
                        subj_fold = subj_folds[0]
                        dataset[subj_fold] = MouseDescriptor(subj.cohort,subj.gender,subj.date,
                                                             subj_fold=subj_fold,cohort_fold=fold_,
                                                             preprocessed=has_head(subj_fold),subj_id=subj.specimen)
                        print("Found and processed")
                    if len(subj_folds) > 1:
                        print(subj_fold)
            else:
                print("pattern error? "  + res + " fn " + fn_)
    f = open("processed_files.pkl","wb")
    pickle.dump(dataset,f)
    f.close()
from configparser import ConfigParser
from typing import Union, List, Dict
from PIL import Image
import SimpleITK as sitk
# deve
def get_subjects(fold : Union[str,pathlib.Path]) -> List[pathlib.Path]:
    """
    parse folder and get all mice folders inside
    :param fold: folder with cohort
    :return: folder with mouse
    """
    i_f = fold
    if isinstance(fold,str):
        i_f = pathlib.Path(i_f)
    res = []
    for file_ in i_f.iterdir():
        if file_.is_file():
            continue
        #filter
        if file_.name.startswith('BMD_TMD') or file_.name.startswith('Morphology') or file_.name.startswith('phantom'):
            continue
        res.append(file_)
    return res


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


def convert_head_folder(fold : Union[str,pathlib.Path]):
    """

    :param fold: folder with a head
    :return:
    """
    i_f = fold
    if isinstance(fold,str):
        i_f = pathlib.Path(i_f)
    #folder already preprocessed
    rec_folder = [ x for x in i_f.iterdir() if x.is_dir() and x.name.endswith("Rec")][0]

    head_folder = list(rec_folder.glob("**/*__head"))[0]

    config_file = [x for x in head_folder.iterdir() if x.name.endswith(".log")][0]
    config = ImageProperties(config_file)

    #image files
    dataset = {}
    for image_file in head_folder.glob("**/*.bmp"):
        #extra filter of head folder
        if image_file.name.endswith("spr.bmp"):
            continue
        ids = (int(image_file.name.replace(".bmp",'').split('rec')[-1]))
        arr = np.array(Image.open(image_file))
        dataset[ids] = arr

    return convert_dataset_to_image(dataset,config)


    #for
    #print(config.image_pixel_size)


MICE_FOLDERS = {PROCESS_DIR / x.subj_fold.name: x for _,x in dataset.items() if x.preprocessed}

### create subfolders
for fold_ in MICE_FOLDERS.keys():
    fold_.mkdir(exist_ok=True)


rule all:
    input:
        expand("{dirname}/ct_image_.nii.gz",dirname=MICE_FOLDERS.keys())


rule convert_images:
    input: "{dirname}"
    output: "{dirname}/ct_image_.nii.gz"

    threads: 1

    run:
        inp_file = pathlib.Path(input[0])
        out_file = pathlib.Path(output[0])
        print(output[0])
        print(str(MICE_FOLDERS[inp_file].subj_fold))
        out = convert_head_folder(str(MICE_FOLDERS[inp_file].subj_fold))
        print("image created")
        sitk.WriteImage(out, output[0])