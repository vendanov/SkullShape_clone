import os
from dataclasses import dataclass

import numpy as np
import cv2
import pandas as pd


@dataclass
class TMDBMDResult:
    
    tmd : float
    bmd: float
    

def tmd_bmd(phantom1_path, phantom2_path, body_path):
    """
    Calculate Tissue Mineral Density (TMD) and Bone Mineral Density (BMD) from image data.

    This function processes image data from three different paths: 'phantom1_path', 'phantom2_path', and 'body_path' to calculate
    TMD and BMD values. It reads image files, applies specific image processing operations, and performs mathematical
    calculations to derive these density values.

    Parameters:
    - phantom1_path (str): Path to the directory containing image data related to 'phantom1.'
    - phantom2_path (str): Path to the directory containing image data related to 'phantom2.'
    - body_path (str): Path to the directory containing image data related to 'body.'

    Returns:
    - TMD (float): Tissue Mineral Density.
    - BMD (float): Bone Mineral Density.

    Raises:
    - Exception: If there is no data for analysis (when 'list_coef14' is empty).
    """
    list_coef14 = []
    list_coefphantom1 = []
    list_coefphantom2 = []

    ############# FANTOM1 Coef ##############
    for file_name in sorted(os.listdir(phantom1_path)):
        if file_name.endswith('.bmp'):
            array_mask = cv2.imread(phantom1_path + '/' + file_name, cv2.IMREAD_GRAYSCALE)
            coefphantom1 = array_mask[array_mask > 0]
            # print(coefphantom1)
            if coefphantom1.size > 0:
                list_coefphantom1 = np.concatenate((list_coefphantom1, coefphantom1), axis=None)
    ############## FANTOM2 Coef ##############
    for file_name in sorted(os.listdir(phantom2_path)):
        if file_name.endswith('.bmp'):
            array_mask = cv2.imread(phantom2_path + '/' + file_name, cv2.IMREAD_GRAYSCALE)
            coefphantom2 = array_mask[array_mask > 0]
            # print(coefphantom2)
            if coefphantom2.size > 0:
                list_coefphantom2 = np.concatenate((list_coefphantom2, coefphantom2), axis=None)

    ###### BODY Coef14 ##############
    for file_name in sorted(os.listdir(body_path)):
        if file_name.endswith('.bmp'):
            # print(file_name)
            array_mask = cv2.imread(body_path + '/' + file_name, cv2.IMREAD_GRAYSCALE)
            coef14 = array_mask[array_mask >= 13]
            if coef14.size > 0:
                list_coef14 = np.concatenate((list_coef14, coef14), axis=None)

    if len(list_coef14) > 0:
        list_coef14 = list_coef14 + 1

        list_coef30 = list_coef14[list_coef14 >= 30]

        mean_phantom1 = np.mean(list_coefphantom1)
        mean_phantom2 = np.mean(list_coefphantom2)
        mean_coef14 = np.mean(list_coef14)
        mean_coef30 = np.mean(list_coef30)

        if mean_phantom1 > mean_phantom2:
            aux = mean_phantom1
            mean_phantom1 = mean_phantom2
            mean_phantom2 = aux

        #####
        list_coef13 = list_coef14[list_coef14 >= 13]
        list_coef31 = list_coef14[list_coef14 >= 31]
        mean_coef13 = np.mean(list_coef13)
        mean_coef31 = np.mean(list_coef31)

        #####
        AUcoef13 = (0.00038235294118 * mean_coef13) + 0.0025
        AUcoef31 = (0.00038235294118 * mean_coef31) + 0.0025
        #######

        AUcoefphantom1 = (0.00038235294118 * mean_phantom1) + 0.0025
        AUcoefphantom2 = (0.00038235294118 * mean_phantom2) + 0.0025
        AUcoef14 = (0.00038235294118 * mean_coef14) + 0.0025
        AUcoef30 = (0.00038235294118 * mean_coef30) + 0.0025

        slope = (AUcoefphantom2 - AUcoefphantom1) / 50
        HAP00 = AUcoefphantom1 - (slope * 25)
        TMD = (AUcoef30 - HAP00) / (100 * slope)
        BMD = (AUcoef14 - HAP00) / (100 * slope)

        ######
        TMD31 = (AUcoef31 - HAP00) / (100 * slope)
        BMD13 = (AUcoef13 - HAP00) / (100 * slope)
        BMD14 = (AUcoef14 - HAP00) / (100 * slope)
        #######

        print("BMD13", "BMD14" "TMD30", "TMD31")
        print(BMD13, BMD14, TMD, TMD31)

        # print(mean_phantom1, mean_phantom2, mean_coef14, mean_coef30)
        print(AUcoefphantom1, AUcoefphantom2, AUcoef14, AUcoef30)
        print("TMD: ", TMD, " BMD: ", BMD)
        return TMD, BMD
    else:
        raise Exception('ERROR!!!! NO Data for Analysis! Look into the folder!')
    pass
