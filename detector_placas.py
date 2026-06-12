"""
detector_placas.py
------------------
Script principal. Dado una foto de un carro:
  1. Carga el modelo entrenado (Keras, PyTorch o TensorFlow)
  2. Predice el bounding box de la placa
  3. Recorta la región de la placa
  4. Aplica EasyOCR para leer los caracteres
  5. Imprime el resultado en pantalla

Uso desde terminal:
    python detector_placas.py --imagen foto_carro.png --modelo keras
    python detector_placas.py --imagen foto_carro.png --modelo pytorch
    python detector_placas.py --imagen foto_carro.png --modelo tensorflow
"""

import os
import argparse
import sys

# Ocultar warnings informativos de TensorFlow para arranque más limpio
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

from utils import IMAGEN_ANCHO, IMAGEN_ALTO, recortar_placa, preprocesar_placa_para_ocr


# Se inicializa solo cuando se necesita, no al arrancar el script
lector_ocr = None


def obtener_lector_ocr():
    """
    Carga EasyOCR la primera vez que se necesita.
    Las siguientes llamadas reutilizan la instancia ya cargada.
    Esto evita que el script tarde 30+ segundos antes de hacer nada útil.
    """
    global lector_ocr

    if lector_ocr is None:
        print("Cargando ")
        import easyocr
        lector_ocr = easyocr.Reader(["en"], gpu=False, verbose=False)

    return lector_ocr


def cargar_imagen_para_modelo(ruta_imagen):
    """
    Carga y preprocesa la imagen para que tenga el formato
    que espera la CNN: (1, 224, 224, 3) con valores en [0, 1].
    El 1 al inicio es el batch_size (una sola imagen).
    """
    imagen_bgr  = cv2.imread(ruta_imagen)

    if imagen_bgr is None:
        print(f"ERROR: No se pudo leer la imagen: {ruta_imagen}")
        sys.exit(1)

    imagen_redim = cv2.resize(imagen_bgr, (IMAGEN_ANCHO, IMAGEN_ALTO))
    imagen_rgb   = cv2.cvtColor(imagen_redim, cv2.COLOR_BGR2RGB)
    imagen_norm  = imagen_rgb.astype(np.float32) / 255.0

    # Agregar dimensión de batch: (224, 224, 3) → (1, 224, 224, 3)
    imagen_batch = np.expand_dims(imagen_norm, axis=0)

    return imagen_bgr, imagen_batch


def predecir_con_keras(ruta_modelo, imagen_batch):
    """
    Carga el modelo de Keras y predice el bounding box.
    Devuelve un array [x_min, y_min, x_max, y_max] normalizado.
    """
    from tensorflow import keras

    modelo       = keras.models.load_model(ruta_modelo)
    prediccion   = modelo.predict(imagen_batch, verbose=0)
    bbox         = prediccion[0]

    return bbox


def predecir_con_pytorch(ruta_modelo, imagen_batch):
    """
    Carga el modelo de PyTorch y predice el bounding box.
    Devuelve un array [x_min, y_min, x_max, y_max] normalizado.
    """
    import torch
    from train_pytorch import CNNDetectorPlacas

    modelo = CNNDetectorPlacas()
    modelo.load_state_dict(torch.load(ruta_modelo, map_location="cpu"))
    modelo.eval()

    # PyTorch espera (batch, canales, alto, ancho)
    imagen_pt = imagen_batch.transpose(0, 3, 1, 2)
    tensor    = torch.tensor(imagen_pt, dtype=torch.float32)

    with torch.no_grad():
        prediccion = modelo(tensor)

    bbox = prediccion[0].numpy()

    return bbox


def predecir_con_tensorflow(ruta_modelo, imagen_batch):
    """
    Carga el modelo de TensorFlow y predice el bounding box.
    Devuelve un array [x_min, y_min, x_max, y_max] normalizado.

    tf.saved_model.load() devuelve un objeto contenedor (_UserObject),
    no el modelo directamente. La funcion de inferencia vive en
    modelo_cargado.__call__ o en las firmas del SavedModel.
    Usamos __call__ que es la forma directa de invocar el forward pass.
    """
    import tensorflow as tf

    modelo_cargado = tf.saved_model.load(ruta_modelo)
    tensor         = tf.constant(imagen_batch, dtype=tf.float32)

    # tf.saved_model.load() guarda las funciones del modelo en
    # modelo_cargado.signatures o en modelo_cargado.f según como
    # fue exportado. Usamos serving_default que es la firma por defecto,
    # o intentamos llamar directamente si está disponible.
    if hasattr(modelo_cargado, "signatures"):
        firma        = modelo_cargado.signatures["serving_default"]
        resultado    = firma(tensor)
        # La firma devuelve un dict; tomamos el primer valor
        prediccion   = list(resultado.values())[0]
    else:
        prediccion   = modelo_cargado(tensor, training=False)

    bbox = prediccion[0].numpy()

    return bbox


def leer_placa_con_ocr(recorte_bgr):
    """
    Usa EasyOCR para leer el texto del recorte de la placa.
    Prueba tres versiones del recorte para maximizar la lectura:
      1. Color original (EasyOCR prefiere esto)
      2. Escala de grises
      3. Con aumento de contraste (CLAHE)
    Toma el resultado con más texto encontrado entre las tres.
    """
    ocr = obtener_lector_ocr()

    # Versión 1: color original
    version_color = recorte_bgr

    # Versión 2: escala de grises convertida a BGR para que EasyOCR la acepte
    gris          = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
    version_gris  = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

    # Versión 3: contraste mejorado con CLAHE
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) mejora
    # el contraste localmente sin sobreexponer zonas brillantes
    clahe              = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gris_clahe         = clahe.apply(gris)
    version_contraste  = cv2.cvtColor(gris_clahe, cv2.COLOR_GRAY2BGR)

    versiones = [version_color, version_gris, version_contraste]
    nombres   = ["color", "gris", "contraste"]

    mejor_texto     = ""
    mejor_confianza = 0.0

    for version, nombre in zip(versiones, nombres):

        resultados = ocr.readtext(version, detail=1)

        textos_version    = []
        confianza_version = 0.0

        for (bbox_ocr, texto, confianza) in resultados:
            # Umbral bajo (10%) para no perder lecturas débiles
            if confianza >= 0.10:
                textos_version.append(texto)
                confianza_version += confianza

        texto_version = " ".join(textos_version)

        

        # Elegir la versión con mayor confianza acumulada
        if confianza_version > mejor_confianza:
            mejor_confianza = confianza_version
            mejor_texto     = texto_version

    return mejor_texto


def detectar_placa(ruta_imagen, tipo_modelo):
    """
    Pipeline completo:
    1. Cargar imagen
    2. Predecir bbox con el modelo elegido
    3. Recortar la placa
    4. Leer el texto con OCR
    5. Imprimir el resultado
    """

    # Rutas de los modelos según el tipo
    rutas_modelos = {
        "keras"      : "models/modelo_keras.keras",
        "pytorch"    : "models/modelo_pytorch.pth",
        "tensorflow" : "models/modelo_tensorflow"
    }

    ruta_modelo = rutas_modelos[tipo_modelo]

    if not os.path.exists(ruta_modelo):
        print(f"ERROR: Modelo no encontrado en '{ruta_modelo}'")
        print(f"Primero ejecuta: python train_{tipo_modelo}.py")
        sys.exit(1)

    print(f"Usando modelo : {tipo_modelo}")
    print(f"Imagen        : {ruta_imagen}")
    

    # Paso 1: cargar y preprocesar imagen
    imagen_original, imagen_batch = cargar_imagen_para_modelo(ruta_imagen)

    # Paso 2: predecir bounding box
    print("Detectando placa...")

    if tipo_modelo == "keras":
        bbox = predecir_con_keras(ruta_modelo, imagen_batch)

    elif tipo_modelo == "pytorch":
        bbox = predecir_con_pytorch(ruta_modelo, imagen_batch)

    elif tipo_modelo == "tensorflow":
        bbox = predecir_con_tensorflow(ruta_modelo, imagen_batch)

    

    # Paso 3: recortar la región de la placa
    recorte = recortar_placa(imagen_original, bbox)

    if recorte.size == 0:
        print("ERROR: El recorte de la placa resultó vacío.")
        print("El modelo puede necesitar más entrenamiento.")
        sys.exit(1)

    # Paso 4: preprocesar el recorte para mejorar OCR
    recorte_procesado = preprocesar_placa_para_ocr(recorte)

    # Guardar el recorte para inspección visual
    # Abre debug_placa.png para verificar que el encuadre sea correcto
    from utils import guardar_recorte_debug
    guardar_recorte_debug(recorte_procesado)

    # Paso 5: leer texto con EasyOCR
    print("Leyendo caracteres de la placa...")
    texto_placa = leer_placa_con_ocr(recorte_procesado)

    # Paso 6: mostrar resultado
    
    if texto_placa:
        print(f"PLACA DETECTADA: {texto_placa}")
    else:
        print("No se pudo leer texto en la placa.")
        print("Intenta con otra imagen o revisa el entrenamiento.")


def main():
    parser = argparse.ArgumentParser(
        description="Detecta y lee la placa de un carro en una imagen."
    )

    parser.add_argument(
        "--imagen",
        type=str,
        required=True,
        help="Ruta a la foto del carro (ej: foto.png)"
    )

    parser.add_argument(
        "--modelo",
        type=str,
        choices=["keras", "pytorch", "tensorflow"],
        required=True,
        help="Librería del modelo a usar: keras, pytorch o tensorflow"
    )

    args = parser.parse_args()

    detectar_placa(
        ruta_imagen  = args.imagen,
        tipo_modelo  = args.modelo
    )


if __name__ == "__main__":
    main()