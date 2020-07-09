"""
Apply filters to PIL.image
"""

import logging
import cv2
import numpy as np
import img_helper

import argparse

from PIL import ImageQt
from PIL import Image

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

import cvui
logger = logging.getLogger()

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-k', '--num-clusters', default=3, type=int,
                             help='Número de clusters para K-Means (por defecto = 3).')
arguments = vars(argument_parser.parse_args())


class ColorFilters:
    filters = { "negative": "Negative", 
                "sepia": "Sepia", 
                "black_white": "Black & White", 
                "Blur": "Blur",
                "Enfoque": "Enfoque",
                "Border": "Border",
                "Noise":"Noise",
                "Poster":"Poster",
                "Portrait": "Portrait",
                "color": "color",
                "intercambio":"intercambio" }
    NEGATIVE, SEPIA, BLACK_WHITE, BLUR, ENFOQUE, BORDES, NOISE, POSTER, PORTRAIT, COLOR,INTERCAMBIO = filters.keys()

def verify_alpha(frame):
    try:
        frame.shape[3]
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    return frame

def alpha_blend(f1, f2, mask):
    alpha = mask/255.0
    blended = cv2.convertScaleAbs(f1 * (1-alpha) + f2 * alpha)
    return blended

def sepia(img, intensity=0.5):
    img = verify_alpha(img)
    h, w, c = img.shape   #alto, ancho, canal

    #configuracion de color
    azul = 20
    verde = 66
    rojo = 112

    sepia_bgra = (azul, verde, rojo, 1)
    overlay = np.full((h, w, 4), sepia_bgra, dtype='uint8')
    cv2.addWeighted(overlay, intensity, img, 1.0, 0, img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return retornar(img)

def blanco_y_negro(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return retornar(img)

def negative(img):
    img = cv2.bitwise_not(img)
    return retornar(img)

def blur(img):
    h, w, c = img.shape
    kernel = np.array([
        [1, 4, 6, 4, 1], 
        [4, 16, 24, 16, 4], 
        [6, 24, 36, 24, 6], 
        [4, 16, 24, 16, 4], 
        [1, 4, 6, 4, 1]],dtype="float")
    kernel*= (1/256)
    img = cv2.filter2D(img, -1, kernel)
    return retornar(img)

def enfoque(img):
    Kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
        ],dtype="float")
    img = cv2.filter2D(img, -1, Kernel)
    return retornar(img)

def bordes(img):
    Kernel = np.ones( (3, 3), dtype="float")
    Kernel*=-1
    Kernel[1][1]=8
    img = cv2.filter2D(img, -1, Kernel)
    return retornar(img)

def portrait(frame):
    frame = verify_alpha(frame)

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_red = np.array([0, 0, 0])
    high_red = np.array([50, 255, 255])
    mask = cv2.inRange(hsv_img, low_red, high_red)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)

    blured = cv2.GaussianBlur(frame, (21, 21), 11)
    blended = alpha_blend(frame, blured, 255-mask)
    frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)

    return retornar(frame)

def color(frame):
    frame = verify_alpha(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_red = np.array([0, 0, 0], np.uint8)
    high_red = np.array([50, 255, 255], np.uint8)

    mask = cv2.inRange(hsv_img, low_red, high_red)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)

    blended = alpha_blend(frame, gray, 255-mask)
    frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)

    return retornar(frame)

def poster(img):
    
    image_copy = np.copy(img)

    pixel_values = image_copy.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    number_of_attempts = 10
    centroid_initialization_strategy = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(pixel_values,
                                arguments['num_clusters'],
                                None,
                                stop_criteria,
                                number_of_attempts,
                                centroid_initialization_strategy)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image_copy.shape)

    return retornar(segmented_image)

def noise(image, sigma=64):
    img = np.array(image)
    noise = np.random.randn(img.shape[0], img.shape[1], img.shape[2])
    img = img.astype('int16')
    img_noise = img + noise * sigma
    img_noise = np.clip(img_noise, 0, 255)
    img_noise = img_noise.astype('uint8')

    return retornar(img_noise)

def nothing(x):
    print(x)

def intercambio(img):
    h,w,c = np.shape(img)
    img2 = img
    for i in range(h):
        for j in range(w):
            a,b,c = img[i][j]
            img2[i][j] = np.array([b,c,a])
    return retornar(img2)

def retornar(cvImg):
    H, W, C = cvImg.shape
    bytesPerLine = 3 * W
    pix = QImage(cvImg.data, W, H, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
    img2 = ImageQt.fromqpixmap(pix)
    return img2

#---------------------------------------------------------------
#-------------------------- Ya no editable ----------------------------
#---------------------------------------------------------------

def color_filter(img, filter_name):
    cvImg = img.copy()
    if filter_name == ColorFilters.NEGATIVE:
        return negative(cvImg)
    elif filter_name == ColorFilters.SEPIA:
        return sepia(cvImg)
    elif filter_name == ColorFilters.BLACK_WHITE:
        return blanco_y_negro(cvImg)
    elif filter_name == ColorFilters.BLUR:
        return blur(cvImg)
    elif filter_name == ColorFilters.ENFOQUE:
        return enfoque(cvImg)
    elif filter_name == ColorFilters.BORDES:
        return bordes(cvImg)
    elif filter_name == ColorFilters.PORTRAIT:
        return portrait(cvImg)
    elif filter_name == ColorFilters.COLOR:
        return color(cvImg)
    elif filter_name == ColorFilters.POSTER:
        return poster(cvImg)
    elif filter_name == ColorFilters.NOISE:
        return noise(cvImg)
    elif filter_name == ColorFilters.INTERCAMBIO:
        return intercambio(cvImg)
    else:
        logger.error(f"can't find filter {filter_name}")
        return img
        # raise ValueError(f"can't find filter {filter_name}")

    # return img