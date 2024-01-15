import os.path
import sys
from pathlib import Path
import pickle
import re
from configparser import ConfigParser
from dataclasses import dataclass
from typing import Union, List, Dict, Tuple
import tqdm.auto
import SimpleITK as sitk
import numpy as np
import pandas as pd
from PIL import Image



cohort_list = (config['cohort_csv'])
#cohort_list = r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/XRY normal skull WT animal.xlsx"
#cohort_list = r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/sel_ko.csv"
input_folders = "/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/"
PROCESS_DIR = Path('/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_ablormal')

print(PROCESS_DIR)
@dataclass
class MouseDescriptor:

    cohort_id: str
    gender: str
    date: str
    subj_id: str
    subj_fold: Path
    cohort_fold: Path
    preprocessed: bool

class Subject:

    def __init__(self,subf):
        self.specimen = subf['SPECIMEN_ID']
        self.cohort = subf['COHORT_ID']
        self.gender = str(subf['GENDER'])
        self.date = subf['DATE']
        if 'Machine' in subf.index: # if machine exists
            self.scanner = subf['Machine']
        else:
            self.scanner = 'N/A'


dataset : Dict[str, MouseDescriptor] = {}

def has_head(path: Path,treshold=300) -> bool:
    """
    return false if no head or head is small
    :param path: path of folder
    :return: True if folder have been processed
    """
    x = list(path.glob("**/*_head"))
    if len(x) > 0:
        head_folder = x[0]
        if len(list(head_folder.glob("**/*.bmp") )) > treshold:
            return True
        else:
            return False
    return False

if cohort_list.endswith('xslx'):
    cohorts = list( pd.read_excel(cohort_list) ['COHORT_ID'])
    cohort_exel = pd.read_excel(cohort_list)
else:
    cohorts = list( pd.read_csv(cohort_list) ['COHORT_ID'])
    cohort_exel = pd.read_csv(cohort_list)
print("Starting processing files")
if os.path.exists("processed_files.pkl"):
    f = open("processed_files.pkl","rb")
    dataset = pickle.load(f)
    f.close()
else:
    for fold_ in tqdm.tqdm(list(Path(input_folders).iterdir())):
    #for fold_ in [Path(r"/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20170925_Klhl5 & Trabd2 & WT(C57Bl-6NCrl)")]:
        #print(str(fold_))
        if fold_.is_file():
            continue
        fn_ =fold_.name.split("_")[0] # First part of fold name
        #print(fn_)
        match = re.search(r'c\d+', fn_)
        if match:
            res = match.group()
            #print(res)
            if res in cohorts:
                indexes = list(cohort_exel[cohort_exel['COHORT_ID'] == res].index)
                ####generate Descriptors
                folers = list(fold_.iterdir()) # folders inside cohort
                for id in indexes:
                    subj = Subject(cohort_exel.loc[id])
                    #get folder with subject
                    #print(cohort_exel.loc[id])
                    #print(fold_)
                    #print("processing cohort " + subj.cohort + " and specimen " + subj.specimen )
                    subj_folds = [x for x in folers if x.name.endswith(subj.specimen)]
                    if len(subj_folds) > 0:
                        subj_fold = subj_folds[0]
                        #print(subj_fold)
                        dataset[subj_fold] = MouseDescriptor(subj.cohort,subj.gender,subj.date,
                                                             subj_fold=subj_fold,cohort_fold=fold_,
                                                             preprocessed=has_head(subj_fold),subj_id=subj.specimen)
                        if "Trabd" in subj_fold.name:
                            print(subj_fold)
                            print(has_head(subj_fold))

                    #    print("Found and processed")
                    #if len(subj_folds) > 1:
                        #print(subj_fold)
            #else:
                #print("pattern error? "  + res + " fn " + fn_)
        else:
            #print("OLDER COHORT ",fn_)
            files = list(fold_.iterdir())
            pattern = re.compile(r".*_.{1}_.*")

            tmp_old_cohorts: List[Tuple[Path, str]] = [(x, x.name) for x in fold_.iterdir()]
            for old_f, o_name in tmp_old_cohorts:
                cohort_name = o_name.split("_")[-1]
                indexes = list(cohort_exel[cohort_exel['SPECIMEN_ID'] == cohort_name].index)
                #print("id",indexes)
                if len(indexes) > 0:
                    index = indexes[0]
                    subj = Subject(cohort_exel.loc[index])
                    #print("processing cohort " + subj.cohort + " and specimen " + subj.specimen)

                    dataset[str(old_f)] = MouseDescriptor(subj.cohort,subj.gender,subj.date,
                        subj_fold=old_f,cohort_fold=fold_,
                        preprocessed=has_head(old_f),subj_id=subj.specimen)
                    #print("Found and processed")
    print(dataset)
    f = open("processed_files.pkl","wb")
    pickle.dump(dataset,f)
    f.close()

# deve
def get_subjects(fold : Union[str,Path]) -> List[Path]:
    """
    parse folder and get all mice folders inside
    :param fold: folder with cohort
    :return: folder with mouse
    """
    i_f = fold
    if isinstance(fold,str):
        i_f = Path(i_f)
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


def convert_head_folder(fold : Union[str,Path]):
    """

    :param fold: folder with a head
    :return:
    """
    i_f = fold
    if isinstance(fold,str):
        i_f = Path(i_f)
    #folder already preprocessed
    rec_folder = [ x for x in i_f.iterdir() if x.is_dir() and x.name.endswith("Rec")][0]

    head_folder = list(rec_folder.glob("**/*_head"))[0]

    config_files = [x for x in head_folder.iterdir() if x.name.endswith(".log")]
    if len(config_files) == 0: # for manual files
        config_files = [x for x in head_folder.parent.iterdir() if x.name.endswith(".log")]
    config_file = config_files[0]
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

print(dataset)
MICE_FOLDERS = {PROCESS_DIR / x.subj_fold.name: x for _,x
                in dataset.items()
                if (x.preprocessed
                    and not (PROCESS_DIR / x.subj_fold.name/"ct_image_.nii.gz").exists()
                    )}



#print([(x.subj_fold.name) for _,x in MICE_FOLDERS.items()])
#sys.exit()
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
        inp_file = Path(input[0])
        out_file = Path(output[0])
        print(output[0])
        print(str(MICE_FOLDERS[inp_file].subj_fold))
        out = convert_head_folder(str(MICE_FOLDERS[inp_file].subj_fold))
        print("image created")
        sitk.WriteImage(out, output[0])