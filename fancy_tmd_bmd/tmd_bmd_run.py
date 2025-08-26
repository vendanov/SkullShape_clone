import os
import glob
from PIL import Image
import shutil
import numpy as np
import cv2
import pandas as pd
import warnings
import shutil

warnings.simplefilter("ignore")
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import model_from_json
import glob, os.path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from IPython.display import clear_output
import random, re, math
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import zipfile
from tensorflow.keras.models import model_from_json
from datetime import datetime
import mysql.connector
import threading
import time
import sys
warnings.simplefilter("ignore")
import os.path
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from tmd_bmd import *

# from tensorflow.keras.applications.vgg16 import VGG16
cohort_folder = sys.argv[1]
cohort_number = cohort_folder.split('c')[2]  # Change here in case your username does not have a 'c'
cohort_number = cohort_number.split('_')[0]
cohort_number = 'c' + cohort_number
print('Cohort folder:', cohort_folder)
print('Cohort number:', cohort_number)

user_name = "varhai"


os.chdir(f'/home/{user_name}/PRIMUS/data/83_BIOINFORMATICS/Julia/Python')
# NN_model_classify_head_slice1


json_file_head_slice = open('NN_model_classify_head_slice1.json', 'r')
loaded_model_json_head_slice = json_file_head_slice.read()
json_file_head_slice.close()

loaded_model_head_slice = model_from_json(loaded_model_json_head_slice)
loaded_model_head_slice.load_weights("NN_model_classify_head_slice1.h5")
print("Model for head slice classification was loadded!")

json_file_masking = open('NN_model_masking_final.json', 'r')
loaded_model_json_masking = json_file_masking.read()
json_file_masking.close()
loaded_model_masking = model_from_json(loaded_model_json_masking)
# load weights into new model
loaded_model_masking.load_weights("NN_model_masking_final.h5")
print("Model for masking was loadded!")

json_file_head_slice_1176 = open('NN_model_classify_head_1176_retrained.json', 'r')
loaded_model_json_head_slice_1176 = json_file_head_slice_1176.read()
json_file_head_slice_1176.close()

loaded_model_head_slice_1176 = model_from_json(loaded_model_json_head_slice_1176)
loaded_model_head_slice_1176.load_weights("NN_model_classify_head_1176_retrained.h5")
print("Model for head slice 1176 classification was loadded!")

json_file_masking_1176 = open('NN_model_masking_1176.json', 'r')
loaded_model_json_masking_1176 = json_file_masking_1176.read()
json_file_masking_1176.close()
loaded_model_masking_1176 = model_from_json(loaded_model_json_masking_1176)
# load weights into new model
loaded_model_masking_1176.load_weights("NN_model_masking_1176.h5")
print("Model for masking 1176 was loadded!")

json_file_holder_masking_1176 = open('NN_model_masking_right_holder_1176.json', 'r')
loaded_model_json_holder_masking_1176 = json_file_holder_masking_1176.read()
json_file_holder_masking_1176.close()
loaded_model_holder_masking_1176 = model_from_json(loaded_model_json_holder_masking_1176)
# load weights into new model
loaded_model_holder_masking_1176.load_weights("NN_model_masking_right_holder_1176.h5")
print("Model for holder masking was loadded!")

# Configuration
IMAGE_SIZE = [256, 256]
EPOCHS = 40
SEED = 777
BATCH_SIZE = 16


def creat_dataframe(img_path):
    os.chdir(img_path)
    thisdir = os.getcwd()
    im_paths = []
    im_ids = []
    for r, d, f in os.walk(thisdir):
        for file in f:
            if file.endswith('.bmp'):
                path = os.path.join(r, file)
                im_paths.append(path)
                im_id = file.split(".")[0]
                im_ids.append(im_id)

    df = pd.DataFrame(data={"id": im_ids, "img_path": im_paths})
    return df


def read_image_and_mask(image_path, mask_path=None, resize=IMAGE_SIZE):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, dtype=tf.float32) / 255.
    return image


def get_validation_dataset(df):
    validation_dataset = tf.data.Dataset.from_tensor_slices(df["img_path"].values)
    validation_dataset = validation_dataset.map(read_image_and_mask)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    return validation_dataset


def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


def new_old_scanner_check(parent):
    df = pd.DataFrame(columns=['ID', 'SCANNER', 'PATH', 'IMAGE_WIDTH', 'IMAGE_HEIGHT'])
    ID = []
    scanner = []
    path = []
    image_width = []
    image_height = []
    split = []
    id_mouse = ''

    df_died = pd.DataFrame(columns=['ID', 'SCANNER', 'PATH', 'IMAGE_WIDTH', 'IMAGE_HEIGHT'])
    ID_died = []
    scanner_died = []
    path_died = []
    image_width_died = []
    image_height_died = []
    id_mouse_died = ''

    isExist = os.path.exists(cohort_folder + '/' + 'csv_files_check_new_old')
    if not isExist:
        os.mkdir(cohort_folder + '/' + 'csv_files_check_new_old')

    for file_name in sorted(os.listdir(parent)):
        #print("FILE NAME: ", file_name)
        if (Path(parent) / file_name / "Ignore").exists():
            print("BROKE ", file_name)
            continue

        if file_name == 'Ignore':
            continue
        # break if head already extracted
        if (file_name.endswith('body') or file_name.endswith('fantom1')
                or file_name.endswith('fantom2')
                or file_name.endswith('mask')
                or file_name.endswith('head') or file_name.endswith('mask_edited')
                or file_name.endswith('predictions')):
            print("BROKE HERE FOR", file_name)
            continue
        if (('ROI' in file_name) or ('VOI' in file_name) or file_name.endswith('RecDXA') or file_name.endswith(
                'Morphology')):
            continue

        if file_name.endswith('+'):
            id_mouse_died = file_name.split('+')[0]
            ID_died.append(id_mouse_died)
            scanner_died.append("DIED")
            path = "".join((parent, '/', file_name))
            path_died.append(path)
            image_width_died.append(None)
            image_height_died.append(None)

        if len(ID_died) > 0:
            df_died['ID'] = ID_died[:1]
            df_died['SCANNER'] = scanner_died[:1]
            df_died['PATH'] = path_died[:1]
            df_died['IMAGE_WIDTH'] = image_width_died[:1]
            df_died['IMAGE_HEIGHT'] = image_height_died[:1]
            df_died.to_csv(cohort_folder + '/' + 'csv_files_check_new_old' + '/' + id_mouse_died + '+' + '.csv',
                           index=False)

        if file_name.endswith(
                '.bmp') and 'VOI' not in parent and '_without_fantoms' not in parent and 'ROI' not in parent:
            split = file_name.split('_')
            id_mouse = file_name.split('rec')[0]
            id_mouse = id_mouse.split('__')[0]
            if id_mouse in ID:  # igor. added this to ignore unnecesary reads
                continue
            im = cv2.imread(parent + '/' + file_name)
            h, w, c = im.shape

            image_width.append(w)
            image_height.append(h)
            # print(id_mouse)
            if ("IR" in split):
                ID.append(id_mouse)
                scanner.append(1176)
                path.append(parent)

            else:
                ID.append(id_mouse)
                scanner.append(1278)
                path.append(parent)
        else:
            current_path = "".join((parent, '/', file_name))
            if os.path.isdir(current_path):
                print(current_path)
                new_old_scanner_check(current_path)
            else:
                continue

    if len(ID) > 0:
        df['ID'] = ID[:1]
        df['SCANNER'] = scanner[:1]
        df['PATH'] = path[:1]
        df['IMAGE_WIDTH'] = image_width[:1]
        df['IMAGE_HEIGHT'] = image_height[:1]
        print(df)

    if len(df) > 0:
        df.to_csv(cohort_folder + '/' + 'csv_files_check_new_old' + '/' + id_mouse + '.csv', index=False)


if os.path.exists(cohort_folder + '/' + 'csv_files_check_new_old'):
    shutil.rmtree(cohort_folder + '/' + 'csv_files_check_new_old')
new_old_scanner_check(cohort_folder)
print("done")
# find all csv files
all_files = glob.glob(os.path.join(cohort_folder + '/' + 'csv_files_check_new_old', "*.csv"))
##create one dataframe with all csv files
df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_cohort = pd.concat(df_from_each_file, ignore_index=True)
# create one csv file
df_cohort.to_csv(cohort_folder + '/' + "Check_1278_1176_died.csv", index=False)
# shutil.rmtree(cohort_folder+'/'+'csv_files_check_new_old')
df_cohort = pd.read_csv(cohort_folder + '/' + "Check_1278_1176_died.csv")
df_cohort.sort_values(by=['ID'], inplace=True)
print("Data Frame for the whole cohort", df_cohort)
df_cohort["IMAGE_WIDTH"] = pd.to_numeric(df_cohort["IMAGE_WIDTH"])
df_cohort["IMAGE_HEIGHT"] = pd.to_numeric(df_cohort["IMAGE_HEIGHT"])
df_cohort['SCANNER'] = df_cohort['SCANNER'].astype(str)
#print("Scanners in the cohort", df_cohort['SCANNER'] ==1278)
df_cohort_1278 = df_cohort.loc[(df_cohort['SCANNER'] == '1278')]  # If df_cohort_1278 is empty, change the '' situation
df_cohort_1176 = df_cohort.loc[df_cohort['SCANNER'] == '1176']  # If df_cohort_1176 is empty, change the '' situation
print('Data Frame for Scanner 1176')
print(df_cohort_1176)
print('Data Frame for Scanner 1278')
print(df_cohort_1278)
df_cohort_1176['ID_without'] = df_cohort_1176.ID.str.split('_').str[2]
if len(df_cohort_1176) == 0 and len(df_cohort_1278) == 0:
    print('No data for this cohort')
    exit()


def predict_head_slice_1176(parent):
    df = pd.DataFrame(columns=['Image_name', 'Propability'])
    # df_whole_mouse= pd.DataFrame(columns = ['Image_name'])

    result_names = []
    result_images = []
    values_pred = []
    fantoms2 = []

    log = ''
    log_file = ''
    ########################## READ IMAGES ##########################################
    # iterate over all the files in directory 'parent'
    for file_name in sorted(os.listdir(parent)):
        # to have the ID of the mouse for creating the folder name
        if file_name.endswith("__fantom1") or file_name.endswith("__fantom2") or file_name.endswith(
                "__without_fantoms") or file_name.startswith("VOI") or file_name.startswith(
            "ROI") or file_name.endswith("RecDXA") or file_name.startswith("masks") or file_name.startswith(
            "Morphology") or file_name.startswith("predictions") or file_name.endswith(
            "Figs") or file_name.endswith("bmdtmd") or file_name.startswith("changes_c159"):
            # skip directories
            continue

        # if file_name in ("Ano2_F_E64-G4667","Ano2_F_E64-G4766","Ano2_M_E64-G4666","Ano2_F_E64-G4668","Ano2_F_E64-G4676","Ano2_F_E64-G4679","Ano2_F_E64-G4765","Ano2_F_E64-G4676","Ano2_F_E64-G4773","Ano2_M_E64-G4662","Ano2_M_E64-G4663"):
        # continue

        if file_name.endswith(".bmp") and not file_name.endswith('spr'):
            image = cv2.imread(parent + '/' + file_name)  # Reading the image (OpenCV)
            image = cv2.resize(image, (150,
                                       150))  # Resize the image, Some images are different sizes. (Resizing is very Important)# add to the list

            result_names.append(file_name)
            result_images.append(image)
        else:
            current_path = "".join((parent, '/', file_name))
            if os.path.isdir(current_path):
                print(current_path)
                predict_head_slice_1176(current_path)
            else:
                continue

    # to array
    pred_images = np.array(result_images)

    #################################### PREDICT HEAD ####################################
    for i in range(0, len(pred_images)):
        # i equal one array of one image
        rnd_number = i
        pred_image = np.array([pred_images[rnd_number]])
        # load_NN_model
        pred_prob = loaded_model_head_slice_1176.predict(pred_image).reshape(2)
        # append propaility for class 'head'
        values_pred.append(pred_prob[0])

    df['Image_name'] = result_names
    df['Propability'] = values_pred
    print(df)

    ################################ FIND IMAGE WITH MAX PROPABILITY ###########################
    # find max propability
    df_sorted = df.sort_values(by=['Propability'], ascending=False)

    for i in range(len(df.index)):  # iterate over max values

        max_value = df_sorted.iloc[i]['Propability']
        if max_value != 0:
            print('MAX Propability is ', max_value)
            # index for max propability value
            index_max = df['Propability'][df['Propability'] == max_value].index
            index_sep = index_max.tolist()[0]
            list_for_folder = df['Image_name'].values.tolist()

            t_head_len = len(list_for_folder[:index_sep])
            t_body_len = len(list_for_folder[index_sep:])
            if (t_head_len < 300) or (t_head_len > t_body_len):
                print("HEAD", t_head_len)
                print("BODY", t_body_len)
                continue
            # print(index_max)

            # name of the file with max propability value
            # print(df['Image_name'][index_max])

            name = df['Image_name'][index_max].to_string(index=False)
            # path to the image to display
            print('Path to image:', parent + '/' + name)
            # read the image
            prediction_slice = cv2.imread(parent + '/' + name)
            # show image
            # plt.imshow(prediction_slice)
            # dont remove this line, will not show the image !!!!!!!
            # plt.show()

            # get id of mouse for the path and folders name
            id_mouse = list_for_folder[0].split('rec')[0]

        ###########To save images to the HEAD and BODY folders ###############
        index = index_max.tolist()
        index_sep = index[0]
        print('Index of predicted head slice:', index_sep)

        list_head = list_for_folder[:index_sep]
        list_body = list_for_folder[index_sep:]

        print('Count of head images:', len(list_head))
        print('Count of body images:', len(list_body))

        ###################################### CREATE PATH, FOLDERS TO SAVE ###########################
        log = 'rec.log'
        log_file = id_mouse + log

        path_to_save_head = os.path.join(parent, id_mouse + '_head')
        path_to_save_body = os.path.join(parent, id_mouse + '_body')

        os.mkdir(path_to_save_head)
        os.mkdir(path_to_save_body)
        print('Folders were created')

        # coping images to the new folder
        for imageName in list_head:
            # print(imageName)
            shutil.copy(os.path.join(parent, imageName), path_to_save_head)
            shutil.copy(os.path.join(parent, log_file), path_to_save_head)
        for imageName in list_body:
            # print(imageName)
            shutil.copy(os.path.join(parent, imageName), path_to_save_body)
            shutil.copy(os.path.join(parent, log_file), path_to_save_body)

        print('Images were copied to the new folders')
        break


def masking_predict_1176(parent, width, height):
    df = pd.DataFrame()
    # df_whole_mouse= pd.DataFrame(columns = ['Image_name'])

    result_names = []
    result_images = []
    values_pred = []
    fantoms2 = []

    log = ''
    log_file = ''
    ########################## READ IMAGES ##########################################
    # iterate over all the files in directory 'parent'
    for file_name in sorted(os.listdir(parent)):
        # to have the ID of the mouse for creating the folder name
        if file_name.endswith("__fantom1") or file_name.endswith("__fantom2") or file_name.startswith(
                "Morphology") or file_name.endswith("__without_fantoms") or file_name.startswith(
            "VOI") or file_name.startswith("ROI") or file_name.endswith("RecDXA") or file_name.endswith(
            "Figs") or file_name.endswith("bmdtmd") or file_name.startswith("changes_c159") or file_name.startswith(
            "masks") or file_name.endswith("__head") or file_name.endswith("__mask"):
            # skip directories
            continue

        if file_name.endswith('__body'):

            current_path = "".join((parent, '/', file_name))
            print(current_path)
            df = creat_dataframe(current_path)
            df = df.sort_values(by=['id'])
            df = df.reset_index(drop=True)

            id_mouse = df.id[0].split('rec')[0]
            df = df[df.id != id_mouse + 'rec_spr']
            id_mouse = Path(parent).name.split("_Rec")[0] + "_"
            print("Id mouse", id_mouse)
            val_dataset = get_validation_dataset(df)
            preds = loaded_model_masking_1176.predict(val_dataset)

            path_to_save_mask = "".join((parent, '/', id_mouse, '_mask'))
            print(path_to_save_mask)
            os.mkdir(path_to_save_mask)

            for i in range(len(preds)):
                th, im_th = cv2.threshold(preds[i], 0.1, 255, cv2.THRESH_BINARY)
                im = cv2.resize(im_th, (width, height))
                im = im.astype(np.uint8)
                # print(im.shape)
                cv2.imwrite(path_to_save_mask + '/' + df['id'][i] + '.bmp', im)

                ##############
            path_to_save_mask_edit = "".join((parent, '/', id_mouse, '_mask_edited'))
            os.mkdir(path_to_save_mask_edit)
            print(path_to_save_mask_edit)

            for i in range(len(preds)):
                array_mask = cv2.imread(path_to_save_mask + '/' + df.id[i] + '.bmp', cv2.COLOR_BGR2GRAY)
                # array_mask = cv2.imread('/home/potip/PRIMUS/data/83_BIOINFORMATICS/Julia/cohort_full/predicted/Cops8_M_E64-G2107__rec00001675.bmp')
                contours, hierarchy = cv2.findContours(array_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                array_len = []

                for s in sorted_contours:
                    # find=sorted_contours[i]
                    len_s = len(s)
                    array_len.append(len_s)
                # print(array_len)

                if len(array_len) <= 1:
                    array_mask = array_mask.astype(np.uint8)
                    cv2.imwrite(path_to_save_mask_edit + '/' + df.id[i] + '.bmp', array_mask)

                while len(sorted_contours) > 1:
                    # print(len(sorted_contours))
                    # print(df.id[i])
                    result = cv2.fillConvexPoly(array_mask, np.array(sorted_contours[1], 'int32'), 0)
                    result = np.uint8(result)
                    # print(result.shape)
                    contours, hierarchy = cv2.findContours(array_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    cv2.imwrite(path_to_save_mask_edit + '/' + df.id[i] + '.bmp', result)

                ################

            path_images_body = "".join((parent, '/', file_name))
            path_to_save_predictions = "".join((parent, '/', id_mouse, '_predictions'))
            os.mkdir(path_to_save_predictions)
            print(path_to_save_predictions)

            for i in range(len(preds)):
                image = cv2.imread(path_images_body + '/' + df.id[i] + '.bmp', cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(path_to_save_mask_edit + '/' + df.id[i] + '.bmp', cv2.IMREAD_GRAYSCALE)

                if len(mask) <= 0 or len(image) <= 0:
                    continue
                else:
                    result = cv2.bitwise_and(image, mask)
                    # print(result.shape)
                    cv2.imwrite(path_to_save_predictions + '/' + df.id[i] + '.bmp', result)

            log = 'rec.log'
            log_file = id_mouse + log
            shutil.copy(os.path.join(parent, log_file), path_to_save_predictions)

            # delete mask folder for the first masks
            shutil.rmtree(path_to_save_mask)
            print("Old folder with masks was deleted:", path_to_save_mask)

        else:
            current_path = "".join((parent, '/', file_name))
            if os.path.isdir(current_path):
                print(current_path)
                masking_predict_1176(current_path, width, height)
            else:
                continue


def TMD_BMD_1176(parent):
    df_BMD_TMD = pd.DataFrame()

    list_coef9 = []
    list_coef30 = []
    list_coefholder = []

    list_ID = []
    list_ID_without = []

    for_df_BMD_TMD = []

    file_name = parent.split('/')[-1]

    if file_name.endswith('_Rec'):  # in future '__predictions'
        id_mouse = file_name.split("_Rec")[0]
        print(id_mouse)
        list_ID.append(id_mouse)

        id_mouse_without = id_mouse.split('_')[-1]
        print(id_mouse_without)
        list_ID_without.append(id_mouse_without)

        holder_path = str(list(Path(parent).glob("*__holder_predictions"))[0])
        body_path = str(list(Path(parent).glob("*__predictions"))[0])
        if not os.path.exists(body_path):
            holder_path = "".join((parent, '/', id_mouse, '__holder_predictions'))
            body_path = "".join((parent, '/', id_mouse, '__predictions'))
        print(body_path)
        ###### BODY Coef14 ##############
        for file_name in sorted(os.listdir(body_path)):
            if file_name.endswith('.bmp'):
                # print(file_name)
                array_mask = cv2.imread(body_path + '/' + file_name, cv2.IMREAD_GRAYSCALE)
                coef9 = array_mask[array_mask >= 8]
                if coef9.size > 0:
                    list_coef9 = np.concatenate((list_coef9, coef9), axis=None)

            ############# FANTOM1 Coef ##############
        print(holder_path)
        for file_name in sorted(os.listdir(holder_path)):
            if file_name.endswith('.bmp'):
                array_mask = cv2.imread(holder_path + '/' + file_name, cv2.IMREAD_GRAYSCALE)
                coefholder = array_mask[array_mask > 0]
                # print(coeffantom1)
                if coefholder.size > 0:
                    list_coefholder = np.concatenate((list_coefholder, coefholder), axis=None)

        """ else:
            current_path = "".join((parent,'/', file_name))
            if os.path.isdir(current_path):
                print(current_path)
                TMD_BMD_1176(current_path)
            else:
                continue
                """

    # we need this shift
    # print(len(list_coef14))
    if len(list_coef9) > 0:
        list_coef9 = list_coef9 + 1

        list_coef30 = list_coef9[list_coef9 >= 30]

        mean_holder = np.mean(list_coefholder)
        mean_coef9 = np.mean(list_coef9)
        mean_coef30 = np.mean(list_coef30)

        #####
        list_coef10 = list_coef9[list_coef9 >= 10]
        list_coef11 = list_coef9[list_coef9 >= 11]
        list_coef13 = list_coef9[list_coef9 >= 13]
        list_coef14 = list_coef9[list_coef9 >= 14]
        list_coef31 = list_coef9[list_coef9 >= 31]
        mean_coef10 = np.mean(list_coef10)
        mean_coef11 = np.mean(list_coef11)
        mean_coef13 = np.mean(list_coef13)
        mean_coef14 = np.mean(list_coef14)
        mean_coef31 = np.mean(list_coef31)
        ######

        AUcoefholder = 0.0047 + (0.00046392157 * mean_holder)

        AUcoeffantom1 = -0.0501 + (15.578 * AUcoefholder)
        AUcoeffantom2 = -0.0135 + (5.7053 * AUcoefholder)
        AUcoef9 = 0.0047 + (0.00046392157 * mean_coef9)
        AUcoef30 = 0.0047 + (0.00046392157 * mean_coef30)

        #####
        AUcoef10 = 0.0047 + (0.00046392157 * mean_coef10)
        AUcoef11 = 0.0047 + (0.00046392157 * mean_coef11)
        AUcoef13 = 0.0047 + (0.00046392157 * mean_coef13)
        AUcoef14 = 0.0047 + (0.00046392157 * mean_coef14)
        AUcoef31 = 0.0047 + (0.00046392157 * mean_coef31)
        #######

        # print(mean_holder,AUcoefholder, AUcoeffantom1, AUcoeffantom2)

        slope = (AUcoeffantom1 - AUcoeffantom2) / 50
        HAP00 = AUcoeffantom2 - (slope * 25)
        TMD = (AUcoef30 - HAP00) / (100 * slope)
        BMD = (AUcoef9 - HAP00) / (100 * slope)

        ######
        TMD31 = (AUcoef31 - HAP00) / (100 * slope)
        BMD10 = (AUcoef10 - HAP00) / (100 * slope)
        BMD11 = (AUcoef11 - HAP00) / (100 * slope)
        BMD13 = (AUcoef13 - HAP00) / (100 * slope)
        BMD14 = (AUcoef14 - HAP00) / (100 * slope)
        #######

        # print(mean_fantom1, mean_fantom2, mean_coef14, mean_coef30)
        print(AUcoeffantom1, AUcoeffantom2, AUcoef9, AUcoef30)
        print("BMD9", "BMD10", "BMD11", "BMD13", "BMD14" "TMD30", "TMD31")
        print(BMD, BMD10, BMD11, BMD13, BMD14, TMD, TMD31)

        path_to_tif = parent.split('/' + list_ID[0] + '_Rec')[0]

        for file_name in sorted(os.listdir(path_to_tif))[:10]:
            if file_name.endswith('.tif'):
                created = os.stat(path_to_tif + '/' + file_name).st_ctime
                created = datetime.fromtimestamp(created)
                only_date, only_time = created.date(), created.time()
                print(file_name, only_date)

        if list_ID and BMD and TMD:
            for_df_BMD_TMD.append(list_ID_without[0])
            print(only_date)
            for_df_BMD_TMD.append(only_date)
            for_df_BMD_TMD.append('UCT 0241')
            for_df_BMD_TMD.append(TMD31)  ############################# CHANGE HERE
            for_df_BMD_TMD.append(BMD14)  ############################# CHANGE HERE

            df_BMD_TMD = df_BMD_TMD.append(pd.DataFrame([for_df_BMD_TMD],
                                                        columns=['CCP_mouseID', 'CCP_date', 'CCP_metadata',
                                                                 'CCP_DXA_793_001', 'CCP_DXA_004_001']),
                                           ignore_index=True)
            df_BMD_TMD.to_csv(cohort_folder + '/' + 'BMD_TMD' + '/' + list_ID[0] + '.csv')
            print(parent + '/' + list_ID[0] + '.csv')
    else:
        print('ERROR!!!! NO Data for Analysis! Look into the folder!')


def holder_masking_predict(parent, width, height):
    df = pd.DataFrame()
    # df_whole_mouse= pd.DataFrame(columns = ['Image_name'])

    result_names = []
    result_images = []
    values_pred = []
    fantoms2 = []

    log = ''
    log_file = ''
    ########################## READ IMAGES ##########################################
    # iterate over all the holder_masking_predictfiles in directory 'parent'
    for file_name in sorted(os.listdir(parent)):
        # to have the ID of the mouse for creating the folder name
        if file_name.endswith("__fantom1") or file_name.endswith("__fantom2") or file_name.startswith(
                "Morphology") or file_name.endswith("__without_fantoms") or file_name.startswith(
            "VOI") or file_name.startswith("ROI") or file_name.endswith("RecDXA") or file_name.endswith(
            "Figs") or file_name.endswith("bmdtmd") or file_name.startswith("changes_c159") or file_name.startswith(
            "masks") or file_name.endswith("__head") or file_name.endswith("__mask"):
            # skip directories
            continue

        if file_name.endswith('__body'):

            current_path = "".join((parent, '/', file_name))
            print(current_path)
            df = creat_dataframe(current_path)
            df = df.sort_values(by=['id'])
            df = df.reset_index(drop=True)

            id_mouse = df.id[0].split('rec')[0]
            df = df[df.id != id_mouse + 'rec_spr']

            val_dataset = get_validation_dataset(df)
            preds = loaded_model_holder_masking_1176.predict(val_dataset)
            id_mouse = Path(parent).name.split("_Rec")[0] + "_"
            path_to_save_mask = "".join((parent, '/', id_mouse, '_holder_mask'))
            print(path_to_save_mask)
            os.mkdir(path_to_save_mask)

            for i in range(len(preds)):
                th, im_th = cv2.threshold(preds[i], 0.1, 255, cv2.THRESH_BINARY)
                im = cv2.resize(im_th, (width, height))
                im = im.astype(np.uint8)
                # print(im.shape)
                cv2.imwrite(path_to_save_mask + '/' + df['id'][i] + '.bmp', im)

                ##############
            path_to_save_mask_edit = "".join((parent, '/', id_mouse, '_holder_mask_edited'))
            os.mkdir(path_to_save_mask_edit)
            print(path_to_save_mask_edit)

            for i in range(len(preds)):
                array_mask = cv2.imread(path_to_save_mask + '/' + df.id[i] + '.bmp', cv2.COLOR_BGR2GRAY)
                # array_mask = cv2.imread('/home/potip/PRIMUS/data/83_BIOINFORMATICS/Julia/cohort_full/predicted/Cops8_M_E64-G2107__rec00001675.bmp')
                contours, hierarchy = cv2.findContours(array_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                array_len = []

                for s in sorted_contours:
                    # find=sorted_contours[i]
                    len_s = len(s)
                    array_len.append(len_s)
                # print(array_len)

                if len(array_len) <= 1:
                    array_mask = array_mask.astype(np.uint8)
                    cv2.imwrite(path_to_save_mask_edit + '/' + df.id[i] + '.bmp', array_mask)

                while len(sorted_contours) > 1:
                    # print(len(sorted_contours))
                    # print(df.id[i])
                    result = cv2.fillConvexPoly(array_mask, np.array(sorted_contours[1], 'int32'), 0)
                    result = np.uint8(result)
                    # print(result.shape)
                    contours, hierarchy = cv2.findContours(array_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    cv2.imwrite(path_to_save_mask_edit + '/' + df.id[i] + '.bmp', result)

            ################

            path_images_body = "".join((parent, '/', file_name))
            path_to_save_predictions = "".join((parent, '/', id_mouse, '_holder_predictions'))
            os.mkdir(path_to_save_predictions)
            print(path_to_save_predictions)

            for i in range(len(preds)):
                image = cv2.imread(path_images_body + '/' + df.id[i] + '.bmp', cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(path_to_save_mask_edit + '/' + df.id[i] + '.bmp', cv2.IMREAD_GRAYSCALE)

                if len(mask) <= 0 or len(image) <= 0:
                    continue
                else:
                    result = cv2.bitwise_and(image, mask)
                    # print(result.shape)
                    cv2.imwrite(path_to_save_predictions + '/' + df.id[i] + '.bmp', result)

            log = 'rec.log'
            log_file = id_mouse + log
            shutil.copy(os.path.join(parent, log_file), path_to_save_predictions)

            # delete mask folder for the first masks
            shutil.rmtree(path_to_save_mask)
            print("Old folder with masks was deleted:", path_to_save_mask)

        else:
            current_path = "".join((parent, '/', file_name))
            if os.path.isdir(current_path):
                print(current_path)
                holder_masking_predict(current_path, width, height)
            else:
                continue

def check_fantoms_created(path):

    if len(list(Path(path).glob("*fantom1"))) > 0 and len(list(Path(path).glob("*fantom2"))) > 0:
        fan1 = list(Path(path).glob("*fantom1"))[0]
        fan2 = list(Path(path).glob("*fantom2"))[0]
        if fan1.is_file():
            fan1.unlink(missing_ok=True)
            fan2.unlink(missing_ok=True)
            return False
        return True
    else:
        return False

def check_head_body_split(path):
    if len(list(Path(path).glob("*head"))) > 0 and len(list(Path(path).glob("*body"))) > 0:
        return True
    else:
        return False


def remove_old_1176_head_scans_folders(path : Path):

    for fold in path.iterdir():
        if fold.is_dir():
            if fold.name.endswith('__head'):
                shutil.rmtree(fold)
            elif fold.name.endswith('__body'):
                shutil.rmtree(fold)
            elif fold.name.endswith('__mask'):
                shutil.rmtree(fold)
            elif fold.name.endswith('mask_edited'):
                shutil.rmtree(fold)
            elif fold.name.endswith('_predictions'):
                shutil.rmtree(fold)


def remove_masks( path : Path):
    masks = list(Path(path).glob("*_mask"))
    for mask in masks:
        if mask.is_dir():
            shutil.rmtree(mask)

    masks = list(Path(path).glob("*_mask_edited"))
    for mask in masks:
        if mask.is_dir():
            shutil.rmtree(mask)


    masks = list(Path(path).glob("*_predictions"))
    for mask in masks:
        if mask.is_dir():
            shutil.rmtree(mask)





print('############## Processing of scanner 1176 ###########')
########################################################################################################## Part 1 for 1176: function to predict the head slice ############################################
init = time.time()
for path in df_cohort_1176['PATH']:

    try:

        print('############## Part 1: function to predict the head slice - Running ###########')
        init = time.time()
        print("STARTED PROCESSING ", path)
        remove_old_1176_head_scans_folders(Path(path))
        predict_head_slice_1176(path)

        end = time.time()
        print("############################This step took " + str(end - init) + " seconds###########################")
        print('############## Part 1 - Finished ###########')


    except Exception as e:
        print("error on path ", path)
        print(e)

print('############## Processing of scanner 1176 FINISHED ###########')
########################################################################################################## Part 1 for 1278: function to detect the fantom s##############################################
print('############## Processing of scanner 1278 ###########')
for path in df_cohort_1278['PATH']:

    try:
        print("STARTED PROCESSING ", path)
        if Path(path).name.split("_Rec")[0] in [x.name.split(".csv")[0] for x in Path(cohort_folder + '/' + 'BMD_TMD').glob("*.csv") ]:
            print("ALREADY PROCESSED ", path)
            continue

        isExist = os.path.exists(cohort_folder + '/' + 'csv_files')
        if not isExist:
            os.mkdir(cohort_folder + '/' + 'csv_files')
        init = time.time()
        if not check_fantoms_created(path):
            fant_df = detect_circle(path, cohort_folder, [], [])

        end = time.time()
        print("function to detect the phantoms ############################This step took " + str(
            end - init) + " seconds###########################")
        init = time.time()
        if not check_head_body_split(path):
            predict_head_slice_1278(path, loaded_model_head_slice=loaded_model_head_slice)
        end = time.time()
        print("function head extract ############################This step took " + str(
            end - init) + " seconds###########################")
        init = time.time()

        selected_row = df_cohort_1278.loc[df_cohort_1278['PATH'] == path]
        masking_predict_1278(path, int(selected_row['IMAGE_WIDTH']), int(selected_row['IMAGE_HEIGHT']),
                             loaded_model_masking)

        end = time.time()
        print("function masking ############################This step took " + str(
            end - init) + " seconds###########################")

        init = time.time()

        mask_fantoms(path)

        end = time.time()

        print("function mask_fantoms ############################This step took " + str(
            end - init) + " seconds###########################")
        init = time.time()
        isBMD_TMD = os.path.exists(cohort_folder + '/' + 'BMD_TMD')
        if not isBMD_TMD:
            os.mkdir(cohort_folder + '/' + 'BMD_TMD')
        TMD_BMD_1278(path, cohort_folder)
        end = time.time()

        print("function TMD_BMD_1278 ############################This step took " + str(
            end - init) + " seconds###########################")

        # remove masks as soon as we have BMD and TMD
        remove_masks(Path(path))

        print("REMOVAL OF MASKS DONE")

    except Exception as e:
        print("error on path ", path)
        print(e)


###### save dxa
print("SAVING DXA FILES")

all_files = glob.glob(os.path.join(cohort_folder + '/' + 'BMD_TMD', "*.csv"))  ################# CHANGE HERE
df_from_each_file = (pd.read_csv(f, sep=',', header=0) for f in all_files)
df_merged = pd.concat(df_from_each_file, ignore_index=True)

df_merged.drop(df_merged.filter(regex="Unname"), axis=1, inplace=True)
df_merged = df_merged.sort_values(by='CCP_mouseID')
df_merged = df_merged.reset_index(drop=True)
df_merged.rename(
    columns={'CCP_mouseID': 'Identificator', 'CCP_date': 'CCP_DXA_001', 'CCP_metadata': '', 'CCP_DXA_793_001': '',
             'CCP_DXA_004_001': ''}, inplace=True)
df_merged.loc[-1] = ['CCP_mouseID', 'CCP_date', 'CCP_metadata', 'CCP_DXA_793_001', 'CCP_DXA_004_001']
df_merged.index = df_merged.index + 1
df_merged.loc[-1] = ['MouseID', 'Date', 'MetaDatalink', 'TMD', 'BMD']
df_merged.index = df_merged.index + 1
df_merged = df_merged.sort_index()
# print(df_merged)
df_merged.to_excel(cohort_folder + '/' + 'DXA ' + cohort_number + '.xlsx', index=False, index_label=False)

# connect to the DB
conn = mysql.connector.connect(host='london02.img.local', port=3306, database='vestlontest', user='vestlontest_bi',
                               password='vJPeF5T6a83j$w8N')
cursor = conn.cursor()
sql = "SELECT * FROM MOUSE WHERE COHORT_ID ='%s'" % (cohort_number)
cursor.execute(sql)
database = pd.DataFrame(cursor.fetchall(), columns=['DOB', 'ID', 'SACRIFICE', 'LAB_ID', 'COHORT_ID', 'CAGE_ID'])
cursor.close()

# Read the excel file
sheet1_df = pd.read_excel(cohort_folder + '/' + 'DXA ' + cohort_number + '.xlsx')

# transform DF
excel_transf = sheet1_df[['Identificator', 'CCP_DXA_001']][2:]
excel_transf = excel_transf.reset_index(drop=True)
excel_transf = excel_transf.rename(columns={"Identificator": "ID", 'CCP_DXA_001': 'Folder_date'})

# transform database table
database = database.sort_values(by='ID')
database = database[['ID', 'SACRIFICE']]


# find id's in folder that ends with '+'
died = [x.split("_")[-1][:-1] for x in glob.glob(os.path.join(cohort_folder, "*+"))]
# find ids that are not in excel table
not_in_table = database['ID'][~database['ID'].isin(excel_transf['ID'])].dropna()
not_in_table = not_in_table.tolist()

for idd in not_in_table:
    if idd in died:
        new_row = {'Identificator': idd, 'CCP_DXA_001': '2021-12-13', 'Unnamed: 2': 'UCT 0245',
                   'Unnamed: 3': 'IMPC_PSC_001', 'Unnamed: 4': 'IMPC_PSC_001'}
    else:
        new_row = {'Identificator': idd, 'CCP_DXA_001': '2021-12-13', 'Unnamed: 2': 'UCT 0241',
                   'Unnamed: 3': 'IMPC_PSC_010', 'Unnamed: 4': 'IMPC_PSC_010'}
    sheet1_df = pd.concat([sheet1_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)


# find ids that are not in excel table
died = database['ID'][~database['ID'].isin(excel_transf['ID'])].dropna()
died = died.tolist()


#
# for i in df_cohort_1176['ID_without'].values:
# sheet1_df.loc[sheet1_df['Identificator'] == i, 'Unnamed: 2'] = 'UCT 0241'

# print(sheet1_df)
# add ids with code


sheet1_df = sheet1_df[2:].sort_values(by='Identificator')
sheet1_df = sheet1_df.reset_index()
sheet1_df.pop('index')
sheet1_df.loc[-1] = ['CCP_mouseID', 'CCP_date', 'CCP_metadata', 'CCP_DXA_793_001', 'CCP_DXA_004_001']
sheet1_df.index = sheet1_df.index + 1
sheet1_df.loc[-1] = ['MouseID', 'Date', 'MetaDatalink', 'TMD', 'BMD']
sheet1_df.index = sheet1_df.index + 1
sheet1_df = sheet1_df.sort_index()
sheet1_df = sheet1_df.rename(columns={"Unnamed: 2": " ", 'Unnamed: 3': '', 'Unnamed: 4': ''})

Python_version = ["3.8"]

# Create DataFrame
if len(df_cohort_1176['ID']) > 0:
    sheet2_df = pd.DataFrame({'MetaData': ['CCP_metadata', 'UCT 0241', 'UCT 0245'],
                              'Date equipment last calibrated': ['CCP_DXA_017_001', excel_transf['Folder_date'][0],
                                                                 excel_transf['Folder_date'][0]],
                              'Operator: Rec': ['CCP_DXA_999_000', 'DXA_SD_009', 'DXA_SD_009'],  # DXA_SD_009
                              'Operator: ROI': ['CCP_DXA_016_001', 'DXA_sfw_009', 'DXA_sfw_009'],
                              'Operator: BMD': ['CCP_DXA_996_000', 'DXA_sfw_009', 'DXA_sfw_009'],
                              'Equipment ID': ['CCP_DXA_011_001', '19G14030', '14G08101'],  # 14G08101
                              'Equipment manufacturer': ['CCP_DXA_012_001', 'Bruker', 'Bruker'],  # Bruker
                              'Equipment model': ['CCP_DXA_013_001', 'SkyScan 1176', 'SkyScan 1278'],
                              'Mouse Status': ['CCP_DXA_014_001', 'Anesthetized', 'Anesthetized'],
                              'Anesthesia': ['CCP_DXA_015_001', 'Zoletil+Xylazine', 'Zoletil+Xylazine'],
                              'DXA_Analytical software producer': ['CCP_DXA_993_001', 'Python Software Foundation',
                                                                   'Python Software Foundation'],
                              'DXA_Analytical software': ['CCP_DXA_992_001', 'Python', 'Python'],
                              'DXA_Analytical software version': ['CCP_DXA_991_001', Python_version[0],
                                                                  Python_version[0]]})
    print(sheet1_df)
else:
    sheet2_df = pd.DataFrame({'MetaData': ['CCP_metadata', 'UCT 0245'],
                              'Date equipment last calibrated': ['CCP_DXA_017_001', excel_transf['Folder_date'][0]],
                              'Operator: Rec': ['CCP_DXA_999_000', 'DXA_SD_009'],  # DXA_SD_009
                              'Operator: ROI': ['CCP_DXA_016_001', 'DXA_sfw_009'],
                              'Operator: BMD': ['CCP_DXA_996_000', 'DXA_sfw_009'],
                              'Equipment ID': ['CCP_DXA_011_001', '19G14030'],  # 14G08101
                              'Equipment manufacturer': ['CCP_DXA_012_001', 'Bruker'],  # Bruker
                              'Equipment model': ['CCP_DXA_013_001', 'SkyScan 1278'],
                              'Mouse Status': ['CCP_DXA_014_001', 'Anesthetized'],
                              'Anesthesia': ['CCP_DXA_015_001', 'Zoletil+Xylazine'],
                              'DXA_Analytical software producer': ['CCP_DXA_993_001', 'Python Software Foundation'],
                              'DXA_Analytical software': ['CCP_DXA_992_001', 'Python'],
                              'DXA_Analytical software version': ['CCP_DXA_991_001', Python_version[0]]})
    print(sheet1_df)

with pd.ExcelWriter(
        f'/home/{user_name}/PRIMUS/data/83_IMPC_DATA/Bioimaging/STATISTICS/_FINISHED_ Tabs_' + '/' + 'DXA ' + cohort_number + '.xlsx') as writer:
    sheet1_df.to_excel(writer, sheet_name="Sheet1", index=False, index_label=False)
    sheet2_df.to_excel(writer, sheet_name="Sheet2", index=False, index_label=False)


## Write all fantoms to a same file
all_files = glob.glob(os.path.join(cohort_folder+'/'+'csv_files', "*.csv"))

df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_from_each_file = list(df_from_each_file)
if len(df_from_each_file) == 0:
    print("No fantoms to save")
    exit()
df_merged   = pd.concat(df_from_each_file, ignore_index=True)
df_merged.to_csv(cohort_folder+'/'+"fantoms_count.csv",index=False)
#shutil.rmtree(cohort_folder+'/'+'csv_files')
print("SAVING DXA FILES FINISHED")