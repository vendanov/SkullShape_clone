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
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt
import zipfile
from tensorflow.keras.models import model_from_json
from datetime import datetime
import mysql.connector
import threading
import time


def read_image_and_mask(image_path, mask_path=None, resize=(256, 256)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize(image, resize)
    image = tf.cast(image, dtype=tf.float32) / 255.
    return image


def masking_predict_1176(parent, width, height, loaded_model_masking):
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
            df = create_dataframe(current_path)
            df = df.sort_values(by=['id'])
            df = df.reset_index(drop=True)

            id_mouse = df.id[0].split('rec')[0]
            df = df[df.id != id_mouse + 'rec_spr']

            val_dataset = get_validation_dataset(df)
            preds = loaded_model_masking.predict(val_dataset)

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
                masking_predict_1176(current_path, width, height, loaded_model_masking)
            else:
                continue


def TMD_BMD_1176(parent, cohort_folder):
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

        holder_path = "".join((parent, '/', id_mouse, '__IR', '__holder_predictions'))
        body_path = "".join((parent, '/', id_mouse, '__IR', '__predictions'))
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


def holder_masking_predict(parent, width, height, loaded_model_holder_masking):
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
            df = create_dataframe(current_path)
            df = df.sort_values(by=['id'])
            df = df.reset_index(drop=True)

            id_mouse = df.id[0].split('rec')[0]
            df = df[df.id != id_mouse + 'rec_spr']

            val_dataset = get_validation_dataset(df)
            preds = loaded_model_holder_masking.predict(val_dataset)

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


def predict_head_slice_1176(parent, loaded_model_head_slice):
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
                predict_head_slice_1176(current_path, loaded_model_head_slice)
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
        pred_prob = loaded_model_head_slice.predict(pred_image).reshape(2)
        # append propaility for class 'head'
        values_pred.append(pred_prob[0])

    df['Image_name'] = result_names
    df['Propability'] = values_pred
    print(df)

    ################################ FIND IMAGE WITH MAX PROPABILITY ###########################
    # find max propability
    max_value = max(df['Propability'], default=0)
    if max_value != 0:
        print('MAX Propability is ', max_value)
        # index for max propability value
        index_max = df['Propability'][df['Propability'] == max_value].index
        # print(index_max)

        # name of the file with max propability value
        # print(df['Image_name'][index_max])

        name = df['Image_name'][index_max].to_string(index=False)
        # path to the image to display
        print('Path to image:', parent + '/' + name)
        # read the image
        prediction_slice = cv2.imread(parent + '/' + name)
        # show image
        plt.imshow(prediction_slice)
        # dont remove this line, will not show the image !!!!!!!
        plt.show()

        list_for_folder = df['Image_name'].values.tolist()
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


def get_validation_dataset(df, batch_size=16):
    validation_dataset = tf.data.Dataset.from_tensor_slices(df["img_path"].values)
    validation_dataset = validation_dataset.map(read_image_and_mask)
    validation_dataset = validation_dataset.batch(batch_size)

    return validation_dataset


def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


def create_dataframe(img_path):
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

