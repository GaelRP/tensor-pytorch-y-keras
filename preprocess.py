"""
preprocess.py
-------------
Preprocesamiento del recorte de la placa vehicular antes de la
segmentación de caracteres.

Cada técnica está justificada en su docstring porque la elección
de parámetros no es obvia y tiene impacto directo en la segmentación.

Uso independiente:
    python preprocess.py --imagen recorte_placa.png
"""

import argparse
import cv2
import numpy as np


# Tamaño al que se amplía el recorte para dar más píxeles por carácter
ANCHO_TRABAJO = 480
ALTO_TRABAJO  = 140


def redimensionar(bgr):
    """
    Amplía el recorte a un tamaño de trabajo fijo.

    Por qué: recortes de placas lejanas pueden medir < 60 × 20 px.
    Con tan pocos píxeles, cada carácter ocupa ~5 px de ancho, lo que
    hace que los filtros y la umbralización sean poco fiables.
    Al escalar a 480 × 140 cada carácter ocupa ~40-60 px de ancho,
    suficiente para que cv2.findContours encuentre formas estables.
    """
    return cv2.resize(bgr, (ANCHO_TRABAJO, ALTO_TRABAJO),
                      interpolation=cv2.INTER_CUBIC)


def a_grises(bgr):
    """
    Convierte a escala de grises.

    Por qué: la información de color de una placa (fondo blanco,
    letras negras, o fondo amarillo con letras negras, etc.) no aporta
    nada para distinguir caracteres del fondo. Reducir a un canal
    acelera todos los pasos siguientes y elimina variaciones de color
    debidas a la iluminación.
    """
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def ecualizar_clahe(gris):
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Por qué: la iluminación de una placa rara vez es uniforme: el
    extremo izquierdo puede estar más iluminado que el derecho.
    La ecualización global de histograma compensa de forma uniforme
    y puede sobreexponer zonas ya brillantes.
    CLAHE divide la imagen en tiles (4×4) y ecualiza cada uno por
    separado, con un clipLimit que evita amplificar el ruido en
    zonas de poco contraste.
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    return clahe.apply(gris)


def suavizar(gris):
    """
    Aplica filtro Gaussiano con kernel 3×3.

    Por qué: la binarización de Otsu es sensible al ruido de alta
    frecuencia (sensor, compresión JPEG). Un kernel pequeño (3×3)
    elimina el ruido de un solo píxel sin difuminar los bordes
    entre carácter y fondo, que son la información que importa.
    Un kernel mayor (≥ 7) borraría esos bordes y fusionaría
    caracteres adyacentes.
    """
    return cv2.GaussianBlur(gris, (3, 3), 0)


def binarizar(gris):
    """
    Binarización con umbral de Otsu.

    Por qué: Otsu calcula automáticamente el umbral que maximiza la
    varianza entre las dos clases (fondo / carácter). Esto es
    importante porque el brillo de fondo varía entre imágenes:
    en una placa bien iluminada el fondo es ~220 y en una placa
    con sombra puede ser ~140. Un umbral fijo de 128 fallaría en uno
    de los dos casos. THRESH_BINARY_INV invierte la imagen para que
    los caracteres queden en blanco (255) sobre fondo negro (0),
    que es lo que cv2.findContours necesita.
    """
    _, binaria = cv2.threshold(
        gris, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binaria


def limpiar_morfologia(binaria):
    """
    Cierre morfológico con kernel 2×2.

    Por qué: después de la binarización los caracteres pueden tener
    pequeños huecos internos (el interior de una 'O' o '0' con ruido).
    El cierre (dilatación → erosión) rellena esos huecos sin engrosar
    significativamente los bordes externos, lo que ayuda a que
    cv2.findContours detecte el contorno exterior correcto.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)


def pipeline_completo(bgr):
    """
    Aplica toda la cadena de preprocesado al recorte BGR de la placa.

    Returns:
        dict con claves:
            "redim"    : BGR redimensionado (para mostrar)
            "gris"     : escala de grises
            "clahe"    : después de CLAHE
            "suavizado": después de Gaussian blur
            "binaria"  : binarización Otsu (fondo negro, chars blancos)
            "limpio"   : después de cierre morfológico (imagen final)
    """
    redim    = redimensionar(bgr)
    gris     = a_grises(redim)
    con_clahe = ecualizar_clahe(gris)
    suav     = suavizar(con_clahe)
    bin_     = binarizar(suav)
    limpio   = limpiar_morfologia(bin_)

    return {
        "redim"    : redim,
        "gris"     : gris,
        "clahe"    : con_clahe,
        "suavizado": suav,
        "binaria"  : bin_,
        "limpio"   : limpio,
    }


def visualizar_pasos(pasos, guardar_en=None):
    """
    Genera una imagen con los 6 pasos del preprocesado, una fila por paso.
    Útil para depuración y para el reporte.
    """
    alto_panel = 100
    separador  = 4
    etiquetas  = ["1. Original", "2. Grises", "3. CLAHE",
                  "4. Suavizado", "5. Otsu", "6. Morfología"]
    claves     = ["redim", "gris", "clahe", "suavizado", "binaria", "limpio"]

    paneles = []
    for etiqueta, clave in zip(etiquetas, claves):
        img = pasos[clave].copy()

        # Convertir grises a BGR para poder apilar con la imagen color
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Escalar al alto fijo manteniendo proporción
        h, w = img.shape[:2]
        nuevo_ancho = max(1, int(w * alto_panel / h))
        img = cv2.resize(img, (nuevo_ancho, alto_panel))

        # Etiqueta
        cv2.putText(img, etiqueta, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

        sep = np.full((alto_panel, separador, 3), 220, dtype=np.uint8)
        paneles.append(img)
        paneles.append(sep)

    fila = np.hstack(paneles[:-1])  # quitar el último separador

    if guardar_en:
        cv2.imwrite(guardar_en, fila)
        print(f"Visualización guardada en: {guardar_en}")

    return fila


def main():
    parser = argparse.ArgumentParser(description="Preprocesa un recorte de placa.")
    parser.add_argument("--imagen", required=True, help="Recorte de la placa (BGR)")
    parser.add_argument("--guardar", default="preproceso_pasos.png")
    args = parser.parse_args()

    bgr = cv2.imread(args.imagen)
    if bgr is None:
        print(f"ERROR: No se pudo leer '{args.imagen}'")
        return

    pasos = pipeline_completo(bgr)
    visualizar_pasos(pasos, guardar_en=args.guardar)
    print("Preprocesado completado.")


if __name__ == "__main__":
    main()
