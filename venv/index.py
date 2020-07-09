import sys
import ntpath
import cv2

import color_filter as cf
import img_helper
import img_transformer as it
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5 import QtCore

from functools import partial

from PIL import ImageQt
from PIL import Image

THUMB_SIZE = 100
Tamaño_Boton = (70,40)
Mini_Botones = (30,40)

def retornar(cvImg):
    H, W, C = cvImg.shape
    bytesPerLine = 3 * W
    pix = QImage(cvImg.data, W, H, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
    img2 = ImageQt.fromqpixmap(pix)
    return img2

class ModificationTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        t_reparar = QPushButton("⚡ Reparar")
        t_reparar.setMinimumSize(*Tamaño_Boton)
        t_reparar.clicked.connect(self.on_repartion)

        t_girar_izquierda = QPushButton("🔃 Invertir")
        t_girar_izquierda.setMinimumSize(*Mini_Botones)
        t_girar_izquierda.clicked.connect(self.on_izquierda)

        t_horizontal = QPushButton("🔄 Horizontal")
        t_horizontal.setMinimumSize(*Mini_Botones)
        t_horizontal.clicked.connect(self.on_horizontal)

        t_espejo = QPushButton("♾ espejo")
        t_espejo.setMinimumSize(*Mini_Botones)
        t_espejo.clicked.connect(self.on_espejo)

        t_fotografia = QPushButton("📷 Captura")
        t_fotografia.setMinimumSize(*Mini_Botones)
        t_fotografia.clicked.connect(self.on_fotografia)

        main_botones = QHBoxLayout()
        main_botones.setAlignment(Qt.AlignCenter)
        main_botones.addWidget(t_reparar)
        main_botones.addWidget(t_girar_izquierda)
        main_botones.addWidget(t_horizontal)
        main_botones.addWidget(t_espejo)
        main_botones.addWidget(t_fotografia)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(main_botones)

        self.setLayout(main_layout)

    def on_repartion(self):
        global _cv_img
        global _cv_filter_image

        imagen_opencv = img_helper.img_trasformer(_cv_img, 'reparacion')
        _cv_filter_image =  retornar(imagen_opencv)
        _cv_img = imagen_opencv.copy()
        self.parent.parent.place_preview_img()

    def on_izquierda(self):
        global _cv_img
        global _cv_filter_image

        imagen_opencv = img_helper.img_trasformer(_cv_img, 'al_revez')
        _cv_filter_image =  retornar(imagen_opencv)
        _cv_img = imagen_opencv.copy()
        self.parent.parent.place_preview_img()

    def on_horizontal(self):
        global _cv_img
        global _cv_filter_image

        imagen_opencv = img_helper.img_trasformer(_cv_img, 'horizontal')
        _cv_filter_image =  retornar(imagen_opencv)
        _cv_img = imagen_opencv.copy()
        self.parent.parent.place_preview_img()

    def on_espejo(self):
        global _cv_img
        global _cv_filter_image

        imagen_opencv = img_helper.img_trasformer(_cv_img, 'espejo')
        _cv_filter_image =  retornar(imagen_opencv)
        _cv_img = imagen_opencv.copy()
        self.parent.parent.place_preview_img()

    def on_fotografia(self):
        global _cv_img
        global _cv_filter_image

        imagen_opencv = img_helper.img_trasformer(_cv_img, 'fotografia')
        _cv_filter_image =  retornar(imagen_opencv)
        _cv_img = imagen_opencv.copy()
        self.parent.parent.place_preview_img()

class FiltersTab(QWidget):
    """Color filters widget"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # scroll layout
        self.scroll_layout = QHBoxLayout()
        # scroll widget
        self.scroll_content = QWidget()
        self.scroll_content.setLayout(self.scroll_layout)
        # scrollview
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.scroll_content)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea.setStyleSheet("QScrollArea{min-height: 140px; max-height: 140px}")
        ##### MAIN LAYOUT
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.scrollArea)

        self.add_filter_thumb("none")

        for key, val in cf.ColorFilters.filters.items():
            self.add_filter_thumb(key, val)

        self.setLayout(self.main_layout)

    def add_filter_thumb(self, name, title=""):
        print(f"create lbl thumb for: {name}")

        thumb_lbl = QLabel(".")
        thumb_lbl.name = name

        if name != "none":
            thumb_lbl.setToolTip(f"Filtro: <b>{title}</b>")
        else:
            thumb_lbl.setToolTip('No filter')

        thumb_lbl.setCursor(Qt.PointingHandCursor)
        thumb_lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb_lbl.mousePressEvent = partial(self.on_filter_select, name)

        self.scroll_layout.addWidget(thumb_lbl)

    def on_filter_select(self, filter_name, e):
        global _cv_img
        global _cv_filter_image

        if filter_name != "none":
            _cv_filter_image = img_helper.color_filter(_cv_img, filter_name)
        else:
            _cv_filter_image = _img_original.copy()

        self.parent.parent.place_preview_img()

class ActionTabs(QTabWidget):
    """Action tabs widget"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.filters_tab = FiltersTab(self)
        self.modification_tab = ModificationTab(self)

        self.addTab(self.filters_tab, "Filtros")
        self.addTab(self.modification_tab, "Modificación")

        #self.setMaximumHeight(190)

class MenuInicio(QWidget):
    """Main widget"""

    def __init__(self):
        super().__init__()

        self.file_name = None

        self.img = QLabel("<b> Bienvenido</b><br>"
                              "<div style='margin: 30px 0'><img src='assets/Logo_Meso.png'/></div>"
                              "<b>Cargar imagen para iniciar</b> <span style='color:red'>&#11014;</span>")
        self.img.setAlignment(Qt.AlignCenter)

        ######### BOTONES
        #CARGAR
        self.upload_btn = QPushButton("Cargar")
        self.upload_btn.clicked.connect(self.on_upload)
        #RESETEAR
        # self.reset_btn = QPushButtoS
        #GUARDAR
        self.save_btn = QPushButton("Guardar")
        self.save_btn.clicked.connect(self.on_save)
        self.save_btn.setEnabled(False)

        # NUEVA LAYOUT DE LOS BOTONES
        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        btn_layout.addWidget(self.upload_btn)
        # btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.save_btn)

        ######## TABS
        self.action_tabs = ActionTabs(self)
        self.action_tabs.setVisible(False)


        ######## INTERFAZ LOADER  ########
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.img)
        self.layout.addWidget(self.action_tabs)
        self.layout.addLayout(btn_layout)

        self.setLayout(self.layout)

    ## SUBIR IMAGEN
    def on_upload(self):
        print("upload")
        #img_path = './assets/Logo_Meso.png'
        img_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Abrir Imagen",
            "/Users/Jonhy/Pictures",
            "Images (*.jpg)"
        )
        if img_path:
            self.file_name = ntpath.basename(img_path)
            
            cvImg = cv2.imread(img_path)
            
            H, W, C = cvImg.shape
            bytesPerLine = 3 * W
            pix = QImage(cvImg.data, W, H, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

            px = QPixmap(img_path)
            self.img.setPixmap(px)
            self.img.setScaledContents(True)
            self.action_tabs.setVisible(True)

            ## globales
            global _img_original
            _img_original = ImageQt.fromqpixmap(pix)

            global _cv_img
            _cv_img = cvImg

            global _cv_filter_image
            _cv_filter_image = cvImg

            ##### resize
            cvimg_thumb = cv2.resize(cvImg, (THUMB_SIZE, THUMB_SIZE))
            img_filter_thumb = img_helper.resize(_img_original, THUMB_SIZE, THUMB_SIZE)

            for thumb in self.action_tabs.filters_tab.findChildren(QLabel):
                if thumb.name != "none":
                    img_filter_preview = cf.color_filter(cvimg_thumb, thumb.name)
                else:
                    img_filter_preview = img_filter_thumb

                preview_pix = ImageQt.toqpixmap(img_filter_preview)
                thumb.setPixmap(preview_pix)

            # self.reset_btn.setEnabled(True)
            self.save_btn.setEnabled(True)

    def place_preview_img(self):
        img = _cv_filter_image
        preview_pix = ImageQt.toqpixmap(img)
        self.img.setPixmap(preview_pix)
                
    def on_save(self):
        new_img_path, _ = QFileDialog.getSaveFileName(
            self, 
            "QFileDialog.getSaveFileName()",
            f"ez_pz_{self.file_name}",
            "Images (*.png *.jpg)"
        )
        if new_img_path:
            img = _cv_filter_image
            img.save(new_img_path)

# MAIN
if __name__ == '__main__':
    app = QApplication(sys.argv)

    #iniciar ventana
    win = MenuInicio()
    win.setMinimumSize(600, 500)
    win.setMaximumSize(1920, 1080)
    win.setWindowTitle('Editor con OpenCV')
    #tamaño de la pantalla
    win.resize(1280, 920)
    win.show()

    sys.exit(app.exec_())