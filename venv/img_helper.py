# """
# Image helper
# Main module of the package
# All image operations should go thorough this module
# """

# from PIL import Image, ImageEnhance
# import logging

import color_filter as cf
import img_transformer as it
#import img_sobre as ims
import cv2

# logger = logging.getLogger()

# # constants
# CONTRAST_FACTOR_MAX = 1.5
# CONTRAST_FACTOR_MIN = 0.5

# SHARPNESS_FACTOR_MAX = 3
# SHARPNESS_FACTOR_MIN = -1

# BRIGHTNESS_FACTOR_MAX = 1.5
# BRIGHTNESS_FACTOR_MIN = 0.5


# def get_img(path):
#     """Return PIL.Image object"""

#     if path == "":
#         logger.error("path is empty of has bad format")
#         raise ValueError("path is empty of has bad format")

#     try:
#         return Image.open(path)
#     except Exception:
#         logger.error(f"can't open the file {path}")
#         raise ValueError(f"can't open the file {path}")

def _get_ratio_height(width, height, r_width):
    return int(r_width/width*height)


def _get_ratio_width(width, height, r_height):
    return int(r_height/height*width)
    
def resize(img, width, height):
    """Resize image"""
    return img.resize((width, height))


def color_filter(img, filter_name):
    """Filter image"""
    return cf.color_filter(img, filter_name)

def img_trasformer (img,trans_name):
    return it.img_transformer(img,trans_name)

#def img_sobre (img,trans_name):
 #   return ims.img_transformer(img,trans_name)


def verify_alpha(frame):
    try:
        frame.shape[3]
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    return frame

# def rotate(img, angle):
#     """Rotate image"""

#     return img.rotate(angle, expand=True)



# def brightness(img, factor):
#     """Adjust image brightness form 0.5-2 (1 - original)"""

#     if factor > BRIGHTNESS_FACTOR_MAX or factor < BRIGHTNESS_FACTOR_MIN:
#         raise ValueError("factor should be [0-2]")

#     enhancer = ImageEnhance.Brightness(img)
#     return enhancer.enhance(factor)


# def contrast(img, factor):
#     """Adjust image contrast form 0.5-1.5 (1 - original)"""

#     if factor > CONTRAST_FACTOR_MAX or factor < CONTRAST_FACTOR_MIN:
#         raise ValueError("factor should be [0.5-1.5]")

#     enhancer = ImageEnhance.Contrast(img)
#     return enhancer.enhance(factor)


# def sharpness(img, factor):
#     """Adjust image sharpness form 0-2 (1 - original)"""

#     if factor > SHARPNESS_FACTOR_MAX or factor < SHARPNESS_FACTOR_MIN:
#         raise ValueError("factor should be [0.5-1.5]")

#     enhancer = ImageEnhance.Sharpness(img)
#     return enhancer.enhance(factor)


# def flip_left(img):
#     """Flip left to right"""

#     return img.transpose(Image.FLIP_LEFT_RIGHT)


# def flip_top(img):
#     """Flip top to bottom"""

#     return img.transpose(Image.FLIP_TOP_BOTTOM)


# def save(img, path):
#     """Save image to hard drive"""

#     img.save(path)


# def open_img(img):
#     """
#     Open image in temporary file
#     !use it only for debug!
#     """

#     img.open()
