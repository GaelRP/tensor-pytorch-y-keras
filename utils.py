"""
utils.py
--------
Funciones compartidas para los 3 scripts de entrenamiento.
Se encarga de:
  - Leer los archivos XML del dataset
  - Recortar la región de la placa de la imagen
  - Preprocesar las imágenes para la CNN
"""

import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np


# Tamaño al que se redimensiona cada imagen completa antes de entrar a la CNN
IMAGEN_ANCHO  = 224
IMAGEN_ALTO   = 224

# Tamaño al que se redimensiona el recorte de la placa para OCR
PLACA_ANCHO = 300
PLACA_ALTO  = 100


def leer_xml(ruta_xml):
    """
    Lee un archivo XML en formato Pascal VOC y devuelve
    el bounding box de la placa como (x_min, y_min, x_max, y_max).

    Si el XML tiene varios objetos toma solo el primero,
    que en este dataset siempre es la placa.
    """
    arbol = ET.parse(ruta_xml)
    raiz  = arbol.getroot()

    objeto    = raiz.find("object")
    bndbox    = objeto.find("bndbox")

    x_min = int(bndbox.find("xmin").text)
    y_min = int(bndbox.find("ymin").text)
    x_max = int(bndbox.find("xmax").text)
    y_max = int(bndbox.find("ymax").text)

    return x_min, y_min, x_max, y_max


def cargar_dataset(carpeta_imagenes, carpeta_anotaciones):
    """
    Recorre el dataset y devuelve:
      - imagenes  : array numpy (N, ALTO, ANCHO, 3) con valores en [0, 1]
      - etiquetas : array numpy (N, 4) con bounding boxes normalizados [0, 1]

    La normalización del bounding box se hace dividiendo cada coordenada
    entre el ancho o alto original de la imagen, para que la CNN pueda
    predecir valores entre 0 y 1 sin importar el tamaño original.
    """
    imagenes  = []
    etiquetas = []

    archivos_xml = sorted(os.listdir(carpeta_anotaciones))

    for nombre_xml in archivos_xml:

        if not nombre_xml.endswith(".xml"):
            continue

        ruta_xml = os.path.join(carpeta_anotaciones, nombre_xml)

        # El nombre de la imagen tiene el mismo nombre base que el XML
        nombre_base  = nombre_xml.replace(".xml", "")
        ruta_imagen  = os.path.join(carpeta_imagenes, nombre_base + ".png")

        if not os.path.exists(ruta_imagen):
            # Algunos archivos pueden estar en .jpg
            ruta_imagen = os.path.join(carpeta_imagenes, nombre_base + ".jpg")

        if not os.path.exists(ruta_imagen):
            print(f"Imagen no encontrada para: {nombre_xml}, se omite.")
            continue

        imagen = cv2.imread(ruta_imagen)

        if imagen is None:
            print(f"No se pudo leer la imagen: {ruta_imagen}, se omite.")
            continue

        alto_original  = imagen.shape[0]
        ancho_original = imagen.shape[1]

        x_min, y_min, x_max, y_max = leer_xml(ruta_xml)

        # Normalizar el bounding box a valores entre 0 y 1
        x_min_norm = x_min / ancho_original
        y_min_norm = y_min / alto_original
        x_max_norm = x_max / ancho_original
        y_max_norm = y_max / alto_original

        # Redimensionar la imagen al tamaño fijo para la CNN
        imagen_redim = cv2.resize(imagen, (IMAGEN_ANCHO, IMAGEN_ALTO))

        # Convertir de BGR (OpenCV) a RGB y normalizar a [0, 1]
        imagen_rgb   = cv2.cvtColor(imagen_redim, cv2.COLOR_BGR2RGB)
        imagen_norm  = imagen_rgb.astype(np.float32) / 255.0

        imagenes.append(imagen_norm)
        etiquetas.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])

    imagenes_array  = np.array(imagenes,  dtype=np.float32)
    etiquetas_array = np.array(etiquetas, dtype=np.float32)

    print(f"Dataset cargado: {len(imagenes_array)} imágenes")

    return imagenes_array, etiquetas_array


def recortar_placa(imagen_bgr, bbox_normalizado):
    """
    Recibe la imagen original en BGR y el bounding box predicho
    (ya desnormalizado si es necesario, o todavía normalizado).

    Si bbox_normalizado=True, multiplica las coordenadas por el
    tamaño real de la imagen.

    Devuelve el recorte de la placa como imagen BGR.
    """
    alto  = imagen_bgr.shape[0]
    ancho = imagen_bgr.shape[1]

    x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox_normalizado

    x_min = int(x_min_norm * ancho)
    y_min = int(y_min_norm * alto)
    x_max = int(x_max_norm * ancho)
    y_max = int(y_max_norm * alto)

    # Asegurar que el recorte no salga de los límites de la imagen
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(ancho, x_max)
    y_max = min(alto, y_max)

    recorte = imagen_bgr[y_min:y_max, x_min:x_max]

    return recorte


def preprocesar_placa_para_ocr(recorte_bgr):
    """
    Aplica preprocesamiento al recorte de la placa para mejorar OCR.
    Devuelve el recorte en color (BGR) redimensionado.
    EasyOCR funciona mejor con la imagen en color que binarizada,
    porque sus modelos internos ya manejan variaciones de iluminación.
    """
    # Agrandar el recorte para que EasyOCR tenga más píxeles por carácter
    # Un recorte muy pequeño hace que los caracteres sean ilegibles
    redimensionado = cv2.resize(recorte_bgr, (PLACA_ANCHO, PLACA_ALTO))

    return redimensionado


def guardar_recorte_debug(recorte_bgr, ruta="debug_placa.png"):
    """
    Guarda el recorte de la placa en disco para inspección visual.
    Útil para verificar si el bounding box está bien posicionado.
    """
    cv2.imwrite(ruta, recorte_bgr)
    print(f"Recorte guardado en: {ruta} (abre para verificar el encuadre)")