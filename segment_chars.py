"""
segment_chars.py
----------------
Segmentación de caracteres individuales a partir de la imagen
binarizada de la placa.

Método: contornos externos (cv2.findContours) + filtrado por
tamaño y relación de aspecto para eliminar ruido y marcas
que no son caracteres.

Uso independiente:
    python segment_chars.py --imagen recorte_placa.png
"""

import argparse
import cv2
import numpy as np

from preprocess import pipeline_completo, ANCHO_TRABAJO, ALTO_TRABAJO


# ── Filtros de contorno ─────────────────────────────────────────────────────
# Expresados como fracción del alto de la imagen de trabajo.
# Un carácter de placa ocupa ~40-85% del alto de la zona de texto.
ALTO_MIN_FRACCION  = 0.30   # más bajo → se incluyen manchas pequeñas
ALTO_MAX_FRACCION  = 0.95   # más alto → se incluye el marco de la placa
ANCHO_MIN_FRACCION = 0.03   # ancho mínimo: 3% del alto (evita líneas finas)
RATIO_MAX          = 2.5    # ancho / alto máximo (chars no son muy anchos)

# Mínimo de caracteres esperado; si se encuentran menos se intenta
# una estrategia alternativa antes de rendirse.
MIN_CHARS = 2


def _filtrar_contornos(contornos, alto_img, ancho_img):
    """
    Filtra los contornos por alto, ancho y relación de aspecto.

    Retorna lista de (x, y, w, h) ordenada de izquierda a derecha.
    """
    alto_min  = alto_img * ALTO_MIN_FRACCION
    alto_max  = alto_img * ALTO_MAX_FRACCION
    ancho_min = alto_img * ANCHO_MIN_FRACCION

    candidatos = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)

        # El contorno no debe abarcar casi toda la imagen (marco de la placa)
        if w > ancho_img * 0.85:
            continue

        if h < alto_min or h > alto_max:
            continue

        if w < ancho_min:
            continue

        if w / h > RATIO_MAX:
            continue

        candidatos.append((x, y, w, h))

    # Ordenar de izquierda a derecha
    candidatos.sort(key=lambda c: c[0])
    return candidatos


def _superponer_bbox_chars(imagen_bgr, candidatos):
    """Devuelve copia de la imagen con rectángulos sobre cada carácter."""
    vis = imagen_bgr.copy()
    for i, (x, y, w, h) in enumerate(candidatos):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 1)
        cv2.putText(vis, str(i + 1), (x, max(y - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)
    return vis


def segmentar(bgr_placa):
    """
    Segmenta los caracteres individuales de un recorte BGR de placa.

    Pasos:
        1. Preprocesado completo (escala de grises, CLAHE, blur, Otsu, morfología)
        2. Búsqueda de contornos externos
        3. Filtrado por tamaño y relación de aspecto
        4. Si se encuentran muy pocos caracteres, se intenta con
           umbral adaptativo como alternativa a Otsu
        5. Extracción de crops individuales

    Args:
        bgr_placa : imagen BGR del recorte de la placa (cualquier tamaño)

    Returns:
        crops     : list[np.ndarray] — recortes BGR de cada carácter,
                    ordenados de izquierda a derecha
        bboxes    : list[tuple]      — (x, y, w, h) de cada carácter
        vis       — imagen de la placa con los bboxes dibujados
        pasos     — dict de pasos intermedios de preprocesado
    """
    pasos    = pipeline_completo(bgr_placa)
    redim    = pasos["redim"]
    binaria  = pasos["limpio"]

    alto_img, ancho_img = binaria.shape

    contornos, _ = cv2.findContours(
        binaria.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidatos = _filtrar_contornos(contornos, alto_img, ancho_img)

    # Si Otsu no encontró suficientes caracteres, intentar con umbral adaptativo
    if len(candidatos) < MIN_CHARS:
        adaptativo = cv2.adaptiveThreshold(
            pasos["suavizado"], 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 8
        )
        contornos2, _ = cv2.findContours(
            adaptativo.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidatos2 = _filtrar_contornos(contornos2, alto_img, ancho_img)

        if len(candidatos2) > len(candidatos):
            candidatos = candidatos2
            binaria    = adaptativo

    # Extraer crops de la imagen en color (más útil para EasyOCR)
    crops = []
    for x, y, w, h in candidatos:
        crop = redim[y: y + h, x: x + w]
        if crop.size > 0:
            crops.append(crop)

    vis = _superponer_bbox_chars(redim, candidatos)

    return crops, candidatos, vis, pasos


def visualizar_caracteres(crops, guardar_en=None):
    """
    Genera una tira horizontal con todos los crops de caracteres.
    Útil para verificar que la segmentación fue correcta.
    """
    if not crops:
        print("No se encontraron caracteres para visualizar.")
        return None

    alto_panel = 80
    sep_ancho  = 4
    paneles    = []

    for i, crop in enumerate(crops):
        h, w = crop.shape[:2]
        nuevo_ancho = max(1, int(w * alto_panel / h))
        c = cv2.resize(crop, (nuevo_ancho, alto_panel))

        cv2.putText(c, str(i + 1), (2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

        sep = np.full((alto_panel, sep_ancho, 3), 200, dtype=np.uint8)
        paneles.append(c)
        paneles.append(sep)

    tira = np.hstack(paneles[:-1])

    if guardar_en:
        cv2.imwrite(guardar_en, tira)
        print(f"Caracteres guardados en: {guardar_en}")

    return tira


def main():
    parser = argparse.ArgumentParser(description="Segmenta caracteres de un recorte de placa.")
    parser.add_argument("--imagen",  required=True, help="Recorte de la placa (BGR)")
    parser.add_argument("--guardar", default="chars_segmentados.png")
    args = parser.parse_args()

    bgr = cv2.imread(args.imagen)
    if bgr is None:
        print(f"ERROR: No se pudo leer '{args.imagen}'")
        return

    crops, bboxes, vis, _ = segmentar(bgr)

    print(f"Caracteres encontrados: {len(crops)}")
    for i, (x, y, w, h) in enumerate(bboxes):
        print(f"  [{i+1}] x={x} y={y} w={w} h={h}")

    cv2.imwrite(args.guardar, vis)
    print(f"Visualización guardada en: {args.guardar}")

    if crops:
        tira = visualizar_caracteres(crops, guardar_en="chars_tira.png")


if __name__ == "__main__":
    main()
