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


def read_image_and_mask(image_path, mask_path=None, resize=(256, 256)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize(image, resize)
    image = tf.cast(image, dtype=tf.float32) / 255.
    return image


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


def mask_fantoms(parent: str):
    """

    :param parent: parent folder name
    :param loaded_model_head_slice: neural net model for processing
    :return:
    """
    log = ''
    log_file = ''
    ########################## READ IMAGES ##########################################
    # iterate over all the files in directory 'parent'
    file_name = parent.split('/')[-1]

    if file_name.endswith('_Rec'):  # in future '__predictions'
        current_path = "".join((parent))
        id_mouse = file_name.split("_Rec")[0]

        fantom1_path = "".join((parent, '/', id_mouse, '__fantom1'))
        print(id_mouse)
        print(fantom1_path)
        df = create_dataframe(fantom1_path)
        df = df.sort_values(by=['id'])

        path_to_mask_fantom1 = "".join((fantom1_path, '/', id_mouse, '__fantom1_mask'))
        os.mkdir(path_to_mask_fantom1)

        path_to_ROI_fantom1 = "".join((fantom1_path, '/', id_mouse, '__fantom1_ROI'))
        os.mkdir(path_to_ROI_fantom1)

        for i in range(len(df)):
            image = cv2.imread(fantom1_path + '/' + df.id[i] + '.bmp', cv2.COLOR_BGR2GRAY)
            min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(image)
            blurred = cv2.GaussianBlur(image, (11, 11), 0)
            mask = cv2.threshold(blurred, max_val - 10, 255, cv2.THRESH_BINARY)[1]

            cv2.imwrite(path_to_mask_fantom1 + '/' + df.id[i] + '.bmp', mask)

            image = cv2.imread(fantom1_path + '/' + df.id[i] + '.bmp', cv2.COLOR_BGR2GRAY)
            mask = cv2.imread(path_to_mask_fantom1 + '/' + df.id[i] + '.bmp', cv2.COLOR_BGR2GRAY)
            result = cv2.bitwise_and(image, mask)

            cv2.imwrite(path_to_ROI_fantom1 + '/' + df.id[i] + '.bmp', result)

            log = '__rec.log'
            log_file = id_mouse + log
            shutil.copy(os.path.join(parent, log_file), path_to_ROI_fantom1)

        ########### Fantom 2 ##################
        fantom2_path = "".join((parent, '/', id_mouse, '__fantom2'))
        print(fantom2_path)
        df = create_dataframe(fantom2_path)
        df = df.sort_values(by=['id'])

        path_to_mask_fantom2 = "".join((fantom2_path, '/', id_mouse, '__fantom2_mask'))
        os.mkdir(path_to_mask_fantom2)

        path_to_ROI_fantom2 = "".join((fantom2_path, '/', id_mouse, '__fantom2_ROI'))
        os.mkdir(path_to_ROI_fantom2)

        for i in range(len(df)):
            image = cv2.imread(fantom2_path + '/' + df.id[i] + '.bmp', cv2.COLOR_BGR2GRAY)
            min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(image)
            blurred = cv2.GaussianBlur(image, (11, 11), 0)
            mask = cv2.threshold(blurred, max_val - 10, 255, cv2.THRESH_BINARY)[1]

            cv2.imwrite(path_to_mask_fantom2 + '/' + df.id[i] + '.bmp', mask)

            image = cv2.imread(fantom2_path + '/' + df.id[i] + '.bmp', cv2.COLOR_BGR2GRAY)
            mask = cv2.imread(path_to_mask_fantom2 + '/' + df.id[i] + '.bmp', cv2.COLOR_BGR2GRAY)
            result = cv2.bitwise_and(image, mask)

            cv2.imwrite(path_to_ROI_fantom2 + '/' + df.id[i] + '.bmp', result)
            log = '__rec.log'
            log_file = id_mouse + log
            shutil.copy(os.path.join(parent, log_file), path_to_ROI_fantom2)

        """ 
        else:
            current_path = "".join((parent,'/', file_name))
            if os.path.isdir(current_path):
                print(current_path)
                mask_fantoms(current_path)
            else:
                continue

       """


def predict_head_slice_1278(parent, loaded_model_head_slice):
    """

    :param parent: folder for processing
    :param loaded_model_head_slice: neural net model for predicting of a slice to be a head
    :return:
    """
    df = pd.DataFrame(columns=['Image_name', 'Propability'])
    # df_whole_mouse= pd.DataFrame(columns = ['Image_name'])

    result_names = []
    result_images = []
    values_pred = []
    fantoms2 = []
    gggg = []
    log = ''
    log_file = ''
    ########################## READ IMAGES ##########################################
    # iterate over all the files in directory 'parent'
    for file_name in sorted(os.listdir(parent)):
        # to have the ID of the mouse for creating the folder name
        if file_name.endswith("__fantom1") or file_name.endswith("__fantom2") or file_name.endswith(
                "__without_fantoms") or file_name.endswith("RexDXA") or file_name.startswith(
            "masks") or file_name.startswith("Morphology") or file_name.startswith(
            "predictions") or file_name.endswith("Figs") or file_name.startswith("changes_c159"):
            # skip directories
            continue
        # if you got some error insome folder, you can skip folders that already done here
        # if file_name in ("Ano2_F_E64-G4667","Ano2_F_E64-G4766","Ano2_M_E64-G4666","Ano2_F_E64-G4668","Ano2_F_E64-G4676","Ano2_F_E64-G4679","Ano2_F_E64-G4765","Ano2_F_E64-G4676","Ano2_F_E64-G4773","Ano2_M_E64-G4662","Ano2_M_E64-G4663"):
        # continue
        # if file_name.startswith("Cst6") or file_name.startswith("Tomm6") or file_name.startswith("Vegfd") or file_name.startswith("wt_F_82Z-09") or file_name.startswith("wt_F_82Z-CF749") or file_name.startswith("wt_F_82Z-CF750"):
        # continue

        if file_name.endswith(".bmp"):
            image = cv2.imread(parent + '/' + file_name)  # Reading the image (OpenCV)
            image = cv2.resize(image, (150,
                                       150))  # Resize the image, Some images are different sizes. (Resizing is very Important)# add to the list

            result_names.append(file_name)
            result_images.append(image)
        else:
            current_path = "".join((parent, '/', file_name))
            if os.path.isdir(current_path):
                print(current_path)
                predict_head_slice_1278(current_path, loaded_model_head_slice)
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
    df_sorted = df.sort_values(by=['Propability'], ascending=False)
    for i in range(len(df.index)):  # iterate over max values

        max_value = df_sorted.iloc[i]['Propability']
        if max_value != 0:
            print('MAX Propability is ', max_value)
            # index for max propability value
            index_max = df['Propability'][df['Propability'] == max_value].index
            # print(index_max)
            index_sep = index_max.tolist()[0]
            list_for_folder = df['Image_name'].values.tolist()
            # name of the file with max propability value
            # print(df['Image_name'][index_max])

            name = df['Image_name'][index_max].to_string(index=False)
            # path to the image to display
            print('Path to image:', parent + '/' + name)
            # read the image
            prediction_slice = cv2.imread(parent + '/' + name)
            # show image
            # dont remove this line, will not show the image !!!!!!!


            # get id of mouse for the path and folders name
            id_mouse = list_for_folder[0].split('rec')[0]

            ################# FIND INDEX OF LAST IMAGE OF FANTOM2 ####################
            path_fantom2 = parent + '/' + id_mouse + 'fantom2'
            print('Path to the fantom2 folder:', path_fantom2)
            for last_file in sorted(os.listdir(path_fantom2), reverse=True):
                fantoms2.append(last_file)

            print('END slice fantom2:', fantoms2[0])

            index_fantom2 = df['Image_name'][df['Image_name'] == fantoms2[0]].index
            index_fantom2_list = index_fantom2.tolist()
            index_fantom2_end = index_fantom2_list[0]
            print('Index of last fanrom2 image in the full data frame:', index_fantom2)

            ###########To save images to the HEAD and BODY folders ###############


            print('Index of predicted head slice:', index_sep)

            list_head = list_for_folder[index_sep:]
            list_body = list_for_folder[index_fantom2_end + 180:index_sep]
            t_head_len = len(list_head)
            t_body_len = len(list_body)
            print('Count of head images:', len(list_head))
            print('Count of body images:', len(list_body))
            if (len(list_head) < 300) or (t_head_len > t_body_len):
                print("HEAD", t_head_len)
                print("BODY", t_body_len)
                continue
            ###################################### CREATE PATH, FOLDERS TO SAVE ###########################
            log = 'rec.log'
            log_file = id_mouse + log

            path_to_save_head = os.path.join(parent, id_mouse + 'head')
            path_to_save_body = os.path.join(parent, id_mouse + 'body')

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


def masking_predict_1278(parent, width, height, loaded_model_masking):
    df = pd.DataFrame()
    # df_whole_mouse= pd.DataFrame(columns = ['Image_name'])

    result_names = []
    result_images = []
    values_pred = []
    fantoms2 = []

    log = ''
    log_file = ''
    ########################## READ IMAGES ##########################################
    file_name = parent.split('/')[-1]

    if file_name.endswith('_Rec'):  # in future '__predictions'
        id_mouse = file_name.split('_Rec')[0]
        print(id_mouse)
        body_path = "".join((parent, '/', id_mouse, '__body'))
        print(body_path)
        df = create_dataframe(body_path)
        df = df.sort_values(by=['id'])
        df = df.reset_index(drop=True)

        id_mouse = df.id[0].split('rec')[0]
        df = df[df.id != id_mouse + 'rec_spr']

        val_dataset = get_validation_dataset(df)
        preds = loaded_model_masking.predict(val_dataset)

        path_to_save_mask = "".join((parent, '/', id_mouse, 'mask'))
        print(path_to_save_mask)
        os.mkdir(path_to_save_mask)

        for i in range(len(preds)):
            th, im_th = cv2.threshold(preds[i], 0.1, 255, cv2.THRESH_BINARY)
            im = cv2.resize(im_th, (width, height))
            im = im.astype(np.uint8)
            # print(im.shape)
            cv2.imwrite(path_to_save_mask + '/' + df['id'][i] + '.bmp', im)

            ##############
        path_to_save_mask_edit = "".join((parent, '/', id_mouse, 'mask_edited'))
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

        path_to_save_predictions = "".join((parent, '/', id_mouse, 'predictions'))
        os.mkdir(path_to_save_predictions)
        print(path_to_save_predictions)

        for i in range(len(preds)):
            image = cv2.imread(body_path + '/' + df.id[i] + '.bmp', cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(path_to_save_mask_edit + '/' + df.id[i] + '.bmp', cv2.IMREAD_GRAYSCALE)
            result = cv2.bitwise_and(image, mask)
            # print(result.shape)
            cv2.imwrite(path_to_save_predictions + '/' + df.id[i] + '.bmp', result)

        log = 'rec.log'
        log_file = id_mouse + log
        shutil.copy(os.path.join(parent, log_file), path_to_save_predictions)

        # delete mask folder for the first masks
        shutil.rmtree(path_to_save_mask)
        print("Old folder with masks was deleted:", path_to_save_mask)

        """   
    else:
        current_path = "".join((parent,'/', file_name))
        if os.path.isdir(current_path):
            print(current_path)
            masking_predict_1278(current_path,  width, height)
        else:
            continue

        """


def TMD_BMD_1278(parent, cohort_folder):
    df_BMD_TMD = pd.DataFrame()

    list_coef14 = []
    list_coef30 = []
    list_coeffantom1 = []
    list_coeffantom2 = []
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

        fantom1_path = "".join((parent, '/', id_mouse, '__fantom1', '/', id_mouse, '__fantom1_ROI'))
        fantom2_path = "".join((parent, '/', id_mouse, '__fantom2', '/', id_mouse, '__fantom2_ROI'))
        body_path = "".join((parent, '/', id_mouse, '__predictions'))
        print(body_path)

        ###### BODY Coef14 ##############
        for file_name in sorted(os.listdir(body_path)):
            if file_name.endswith('.bmp'):
                # print(file_name)
                array_mask = cv2.imread(body_path + '/' + file_name, cv2.IMREAD_GRAYSCALE)
                coef14 = array_mask[array_mask >= 13]
                if coef14.size > 0:
                    list_coef14 = np.concatenate((list_coef14, coef14), axis=None)

        ############# FANTOM1 Coef ##############
        for file_name in sorted(os.listdir(fantom1_path)):
            if file_name.endswith('.bmp'):
                array_mask = cv2.imread(fantom1_path + '/' + file_name, cv2.IMREAD_GRAYSCALE)
                coeffantom1 = array_mask[array_mask > 0]
                # print(coeffantom1)
                if coeffantom1.size > 0:
                    list_coeffantom1 = np.concatenate((list_coeffantom1, coeffantom1), axis=None)

        ############## FANTOM2 Coef ##############
        for file_name in sorted(os.listdir(fantom2_path)):
            if file_name.endswith('.bmp'):
                array_mask = cv2.imread(fantom2_path + '/' + file_name, cv2.IMREAD_GRAYSCALE)
                coeffantom2 = array_mask[array_mask > 0]
                # print(coeffantom2)
                if coeffantom2.size > 0:
                    list_coeffantom2 = np.concatenate((list_coeffantom2, coeffantom2), axis=None)

            """           

        else:
            current_path = "".join((parent,'/', file_name))
            if os.path.isdir(current_path):
                print(current_path)
                TMD_BMD_1278(current_path)
            else:
                continue

            """

            # we need this shift
    # print(len(list_coef14))
    if len(list_coef14) > 0:
        list_coef14 = list_coef14 + 1

        list_coef30 = list_coef14[list_coef14 >= 30]

        mean_fantom1 = np.mean(list_coeffantom1)
        mean_fantom2 = np.mean(list_coeffantom2)
        mean_coef14 = np.mean(list_coef14)
        mean_coef30 = np.mean(list_coef30)

        if mean_fantom1 > mean_fantom2:
            aux = mean_fantom1
            mean_fantom1 = mean_fantom2
            mean_fantom2 = aux

        #####
        list_coef13 = list_coef14[list_coef14 >= 13]
        list_coef31 = list_coef14[list_coef14 >= 31]
        mean_coef13 = np.mean(list_coef13)
        mean_coef31 = np.mean(list_coef31)

        #####
        AUcoef13 = (0.00038235294118 * mean_coef13) + 0.0025
        AUcoef31 = (0.00038235294118 * mean_coef31) + 0.0025
        #######

        AUcoeffantom1 = (0.00038235294118 * mean_fantom1) + 0.0025
        AUcoeffantom2 = (0.00038235294118 * mean_fantom2) + 0.0025
        AUcoef14 = (0.00038235294118 * mean_coef14) + 0.0025
        AUcoef30 = (0.00038235294118 * mean_coef30) + 0.0025

        slope = (AUcoeffantom2 - AUcoeffantom1) / 50
        HAP00 = AUcoeffantom1 - (slope * 25)
        TMD = (AUcoef30 - HAP00) / (100 * slope)
        BMD = (AUcoef14 - HAP00) / (100 * slope)

        ######
        TMD31 = (AUcoef31 - HAP00) / (100 * slope)
        BMD13 = (AUcoef13 - HAP00) / (100 * slope)
        BMD14 = (AUcoef14 - HAP00) / (100 * slope)
        #######

        print("BMD13", "BMD14" "TMD30", "TMD31")
        print(BMD13, BMD14, TMD, TMD31)

        # print(mean_fantom1, mean_fantom2, mean_coef14, mean_coef30)
        print(AUcoeffantom1, AUcoeffantom2, AUcoef14, AUcoef30)
        print(BMD, TMD)

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
            for_df_BMD_TMD.append('UCT 0245')
            for_df_BMD_TMD.append(TMD31)
            for_df_BMD_TMD.append(BMD14)  ############### CHANGE HERE

            df_BMD_TMD = df_BMD_TMD.append(pd.DataFrame([for_df_BMD_TMD],
                                                        columns=['CCP_mouseID', 'CCP_date', 'CCP_metadata',
                                                                 'CCP_DXA_793_001', 'CCP_DXA_004_001']),
                                           ignore_index=True)
            # print(AU_fantom1)
            df_BMD_TMD.to_csv(cohort_folder + '/' + 'BMD_TMD' + '/' + list_ID[0] + '.csv')  ########## CHANGE HERE
            print(parent + '/' + list_ID[0] + '.csv')
    else:
        print('ERROR!!!! NO Data for Analysis! Look into the folder!')


def detect_circle(parent, cohort_folder, visited_path, failed_path):
    df = pd.DataFrame(columns=['Image_name', 'Circle'])
    df_xls_fantom = pd.DataFrame(columns=['Image_name', 'Count_fantom1', 'Count_fantom2'])

    result_names = []
    result_circle = []
    result_cilinder = []
    xls_names = []
    xls_fantom1_count = []
    xls_fantom2_count = []
    count1 = 0
    count2 = 0

    log = ''
    log_file = ''

    # iterate over all the files in directory 'parent'
    for file_name in sorted(os.listdir(parent)):
        current_path = "".join((parent, '/', file_name))

        if file_name.find("IR") > 0:
            continue

        # to have the ID of the mouse for creating the folder name
        if file_name.endswith("csv_files") or file_name.startswith("masks") or file_name.startswith(
                "Morphology") or file_name.startswith("predictions"):
            # skip directories
            continue

        if current_path in visited_path:
            continue

        if current_path in failed_path:
            continue

        if file_name.endswith(".bmp"):
            # if it's a txt file, print its name (or do whatever you want)
            bgr_img = cv2.imread(parent + '/' + file_name)
            # to grayscale the images
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.blur(gray, (3, 3))
            # to detect circles
            detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30,
                                                minRadius=6, maxRadius=21)
            # changed the 'minRadius' variable from 12 to 6 to account for images that were half the size. From 848 to 424.
            # add to the list
            result_names.append(file_name)
            result_circle.append(detected_circles)
        else:
            current_path = "".join((parent, '/', file_name))
            if os.path.isdir(current_path):
                print("TESTING NEW1 ", current_path)
                detect_circle(current_path, cohort_folder, visited_path, failed_path)
            else:
                continue

    # lists to dataframe columns
    #print('Names ', result_names)
    df['Image_name'] = result_names[:500]
    df['Circle'] = result_circle[:500]
    print('Circle ',df['Circle'])
    # create a index to calculate the differance
    df['num'] = pd.Series([x + 1 for x in range(0, len(df))])

    # delete rows without detecting
    data = df.dropna(axis=0)
    print("DATA ",data)
    # calculate difference between rows to find cilinder 2 starts
    data['diff'] = data['num'].diff()

    print("DATA2 ",data)
    print("DATA DIF ",data['diff'].values)
    print("DATA NUM ",data['num'].values)
    # to find row with maximum differemce to separate cil1 and cil2 rows
    row_find = data[data['diff'] > 10]

    if row_find.empty:
        print('No fantoms were found!')

    row_find = row_find.sort_values(by=['diff', 'num'], ascending=False)
    print(row_find)

    # print(row_find)

    #  get value for separating
    value_sep = row_find.iloc[0:1]['num']
    print(value_sep)

    # all lists, that are needed in the FOR loop
    df_cilinder_1 = []
    start_index_cilinder_1 = []
    end_index_cilinder_1 = []
    df_cilinder1_save = []

    df_cilinder_2 = []
    start_index_cilinder_2 = []
    end_index_cilinder_2 = []
    df_cilinder2_save = []

    folder_name_split = []

    path_to_save_cilinder1 = []
    path_to_save_cilinder2 = []

    list_of_single_column_cil1 = []
    list_of_single_column_cil2 = []

    for value in value_sep:
        sep = int(value)
        # to create a new column
        data['cilinder'] = np.where(data['num'] < sep, 'cilinder 1', 'cilinder 2')
        # to have only rows with value 'cilinder 1'
        df_cilinder_1 = data[data['cilinder'] == 'cilinder 1']
        # to have only rows with value 'cilinder 2'
        df_cilinder_2 = data[data['cilinder'] == 'cilinder 2']

        # index of start cilinder1
        start_index_cilinder_1 = df_cilinder_1.index[0]
        # index of end cilinder1
        end_index_cilinder_1 = df_cilinder_1.index[-1]

        # index of start cilinder2
        start_index_cilinder_2 = df_cilinder_2.index[0]
        # index of end cilinder2
        end_index_cilinder_2 = df_cilinder_2.index[-1]

        # to have all row, encluded missing row without detected circles, to copy all images cilinder 1 in order to the new folder
        df_cilinder1_save = df[(df['num'] > start_index_cilinder_1 + 5) & (df['num'] <= end_index_cilinder_1 - 4)]
        # to have all row, encluded missing row without detected circles, to copy all images cilinder 2 in order to the new folder
        df_cilinder2_save = df[(df['num'] > start_index_cilinder_2 + 5) & (df['num'] <= end_index_cilinder_2 - 4)]

        # form dataframe to list
        list_of_single_column_cil1 = df_cilinder1_save['Image_name'].tolist()
        list_of_single_column_cil2 = df_cilinder2_save['Image_name'].tolist()

        # to have the ID of the mouse for creating the folder name

        folder_name_split = list_of_single_column_cil1[0].split('rec')[0]
        log = 'rec.log'
        log_file = folder_name_split + log

        # path to the new folder to save only cilinder
        path_to_save_cilinder1 = os.path.join(parent, folder_name_split + 'fantom1')
        path_to_save_cilinder2 = os.path.join(parent, folder_name_split + 'fantom2')

        file_name = file_name.split("_~")[0]
        count1 = len(list_of_single_column_cil1)
        count2 = len(list_of_single_column_cil2)
        if count1 != 0:
            xls_fantom1_count.append(count1)
            xls_fantom2_count.append(count2)
            xls_names.append(file_name)
            df_xls_fantom['Count_fantom1'] = xls_fantom1_count
            df_xls_fantom['Count_fantom2'] = xls_fantom2_count
            df_xls_fantom['Image_name'] = xls_names
            print(df_xls_fantom)

        if xls_fantom1_count[0] >= 100 or xls_fantom2_count[0] >= 100:
            split_rec = os.path.split(current_path)
            mouse_folder = os.path.split(split_rec[0])
            if mouse_folder[0] != cohort_folder and mouse_folder[0] not in visited_path and mouse_folder[
                0] not in failed_path:
                failed_path.append(mouse_folder[0])
                print("Take care about this data, something went wrong!!!!:", failed_path)
                continue

        df_xls_fantom.to_csv(cohort_folder + '/' + 'csv_files' + '/' + folder_name_split + '.csv', index=False)

        # path to the new folder to save only cilinder
        os.mkdir(path_to_save_cilinder1)
        os.mkdir(path_to_save_cilinder2)

    print('Folders were created')

    # coping images to the new folder
    for imageName in list_of_single_column_cil1:
        # print(imageName)
        shutil.copy(os.path.join(parent, imageName), path_to_save_cilinder1)
        shutil.copy(os.path.join(parent, log_file), path_to_save_cilinder1)
    for imageName in list_of_single_column_cil2:
        # print(imageName)
        shutil.copy(os.path.join(parent, imageName), path_to_save_cilinder2)
        shutil.copy(os.path.join(parent, log_file), path_to_save_cilinder2)

    print('Images were copied to the new folders')

    if not row_find.empty:
        split_rec = os.path.split(current_path)
        mouse_folder = os.path.split(split_rec[0])
        if mouse_folder[0] != cohort_folder or mouse_folder[0] != split_rec or mouse_folder[0] not in visited_path or \
                xls_fantom1_count[0] < 100 or xls_fantom2_count[0] < 100 or mouse_folder[0] != '':
            visited_path.append(mouse_folder[0])