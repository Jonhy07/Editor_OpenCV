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

dibujando= False
modo = False
(ix,iy) = (-1,-1)
img = np.array([0])
imagen_temporal = np.array([0])
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-k', '--num-clusters', default=3, type=int,
                             help='Número de clusters para K-Means (por defecto = 3).')
arguments = vars(argument_parser.parse_args())

class L_Reparaciones:
    l_reparaciones = {"reparacion": "Reparar",'al_revez':'al_revez','horizontal':'horizontal','espejo':'espejo','fotografia':'fotografia'
			   }
    REPARAR = l_reparaciones.keys()

def Repara(img2):
	global imagen_temporal
	imagen_temporal = img2
	img2=doble_dibujo()
	return img2

def al_revez(img):
	img2 = cv2.flip(img,0)
	return img2

def der_izq(img):
	img2 = cv2.flip(img,1)
	return img2

def espejo(img):
	h = img.shape[1] // 2
	print('tas Aqui')

	img[:,:h] = cv2.flip(img[:,h:],1)
	return img

def fotografia(img):
	print('Aqui')
	img2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
	h, w, c = np.shape(img2)
	if (h == 120):
		img2 = cv2.imread('F.jpg', 1)
	else:
		img2 = videoCamara()

	resize_img = cv2.resize(img2, (w, h))
	h, w, c = np.shape(resize_img)
	for i in range(h):
		for j in range(w):
			a, b, c = resize_img[i][j]
	return resize_img

def doble_dibujo():
	global modo,dibujando,ix,iy,img,imagen_temporal

	nameFrame="Hi"
	mascara="Mascara"

	h, w, c = np.shape(imagen_temporal)

	img = np.zeros((h,w,3),np.uint8)

	cv2.namedWindow(mascara)
	cv2.imshow(nameFrame, imagen_temporal)

	cv2.setMouseCallback(nameFrame,draw_circle)

	while(True):

		cv2.imshow(nameFrame,imagen_temporal)
		cv2.imshow(mascara,img)

		k = cv2.waitKey(1)
		if k==ord('m') or k==ord('M'):
			if modo == True:
				modo = False
			else:
				modo = True
		if k==27:
			break

	img = cv2.medianBlur(img,3)

	cv2.destroyAllWindows()
	cv2.imwrite('mascaraT.png',img)
	img=cv2.imread('mascaraT.png',cv2.IMREAD_GRAYSCALE)

	restaurada = cv2.inpaint(imagen_temporal,img,3,cv2.INPAINT_TELEA)
	restaurada = cv2.inpaint(restaurada,img,3,cv2.INPAINT_TELEA)
	restaurada = cv2.inpaint(restaurada,img,3,cv2.INPAINT_TELEA)

	cv2.destroyAllWindows()
	return restaurada

def draw_circle(event,x,y,flags,param):
	global ix,iy,dibujando,modo,img,imagen_temporal
	if event == cv2.EVENT_LBUTTONDOWN:
		dibujando=True
		(ix,iy)=x,y
	elif event ==cv2.EVENT_MOUSEMOVE:
		if dibujando==True:
			if modo ==True:
				img=cv2.rectangle(img,(ix,iy),(x,y),(255,255,255),-1)
				imagen_temporal=cv2.rectangle(imagen_temporal,(ix,iy),(x,y),(255,255,255),-1)
			else:
				img=cv2.circle(img,(x,y),5,(255,255,255),-1)
				imagen_temporal=cv2.circle(imagen_temporal,(x,y),5,(255,255,255),-1)
	elif event == cv2.EVENT_LBUTTONUP:
		dibujando=False
		if modo == True:
			img=cv2.rectangle(img,(ix,iy),(x,y),(255,255,255),-1)
			imagen_temporal=cv2.rectangle(imagen_temporal,(ix,iy),(x,y),(255,255,255),-1)
		else:
			img=cv2.circle(img,(x,y),5,(255,255,255),-1)
			imagen_temporal=cv2.circle(imagen_temporal,(x,y),5,(255,255,255),-1)

def videoCamara():
    cap = cv2.VideoCapture(0)
    WINDOW_NAME = 'pytagram'
    while (True):
        ret, frame = cap.read()
        cv2.imshow('Video Camara', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            f = frame
            break
    cap.release()
    cv2.destroyAllWindows()
    return f

def retornar(cvImg):
    H, W, C = cvImg.shape
    bytesPerLine = 3 * W
    pix = QImage(cvImg.data, W, H, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
    img2 = ImageQt.fromqpixmap(pix)
    return img2

def img_transformer(cvImg, transformacion_name):
	if transformacion_name == 'reparacion':
		return Repara(cvImg)
	elif transformacion_name == 'al_revez':
		return al_revez(cvImg)
	elif transformacion_name == 'horizontal':
		return der_izq(cvImg)
	elif transformacion_name == 'espejo':
		return espejo(cvImg)
	elif transformacion_name == 'fotografia':
		return fotografia(cvImg)
	else:
		logger.error(f"can't find filter {filter_name}")
		return img