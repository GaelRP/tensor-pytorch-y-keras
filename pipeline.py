"""
pipeline.py
-----------
Pipeline ALPR integrado:
  1. Detección de la placa con CNN (Keras / PyTorch / TensorFlow)
  2. Extracción del ROI (crop)
  3. Preprocesamiento (preprocess.py)
  4. Segmentación de caracteres (segment_chars.py)
  5. Reconocimiento con EasyOCR
  6. Salida visual: imagen original → bbox → crop → chars → texto

Interfaz principal:
    resultado = pipeline("imagen.jpg")
    resultado = pipeline("imagen.jpg", modelo="pytorch")
    # resultado = {
    #     "placa"     : "ABC1234",
    #     "confianza" : 0.0,          # YOLO-style conf no aplica aquí;
    #                                  # se devuelve confianza OCR acumulada
    #     "bbox"      : [x1,y1,x2,y2] en píxeles absolutos, o None
    #     "n_chars"   : 7,
    #     "imagen_resultado": np.ndarray anotada
    # }

Uso desde terminal:
    python pipeline.py --imagen foto.png
    python pipeline.py --imagen foto.png --modelo pytorch --guardar salida.png
"""

import os
import sys
import time
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

from utils import IMAGEN_ANCHO, IMAGEN_ALTO, recortar_placa
from preprocess import pipeline_completo, visualizar_pasos
from segment_chars import segmentar, visualizar_caracteres


# ── Lazy-loading de modelos y OCR ──────────────────────────────────────────────

_modelos_cache  = {}
_lector_ocr     = None

RUTAS_MODELOS = {
    "keras"      : "models/modelo_keras.keras",
    "pytorch"    : "models/modelo_pytorch.pth",
    "tensorflow" : "models/modelo_tensorflow",
}

MODELO_POR_DEFECTO = "pytorch"


def _cargar_modelo(tipo):
    if tipo in _modelos_cache:
        return _modelos_cache[tipo]

    ruta = RUTAS_MODELOS[tipo]
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"Modelo '{tipo}' no encontrado en '{ruta}'. "
            f"Ejecuta primero: python train_{tipo}.py"
        )

    if tipo == "keras":
        from tensorflow import keras
        m = keras.models.load_model(ruta)

    elif tipo == "pytorch":
        import torch
        from train_pytorch import CNNDetectorPlacas
        m = CNNDetectorPlacas()
        m.load_state_dict(torch.load(ruta, map_location="cpu"))
        m.eval()

    elif tipo == "tensorflow":
        import tensorflow as tf
        m = tf.saved_model.load(ruta)

    _modelos_cache[tipo] = m
    return m


def _obtener_ocr():
    global _lector_ocr
    if _lector_ocr is None:
        print("Cargando EasyOCR...")
        import easyocr
        _lector_ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _lector_ocr


# ── Predicción de bbox ─────────────────────────────────────────────────────────

def _preparar_batch(bgr):
    """Convierte BGR a batch (1, 224, 224, 3) normalizado en [0,1]."""
    redim = cv2.resize(bgr, (IMAGEN_ANCHO, IMAGEN_ALTO))
    rgb   = cv2.cvtColor(redim, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(rgb, axis=0)


def _predecir_bbox(bgr, tipo):
    batch = _preparar_batch(bgr)
    m     = _cargar_modelo(tipo)

    if tipo == "keras":
        return m.predict(batch, verbose=0)[0]

    elif tipo == "pytorch":
        import torch
        pt = torch.tensor(batch.transpose(0, 3, 1, 2), dtype=torch.float32)
        with torch.no_grad():
            return m(pt)[0].numpy()

    elif tipo == "tensorflow":
        import tensorflow as tf
        t = tf.constant(batch, dtype=tf.float32)
        if hasattr(m, "signatures"):
            res = m.signatures["serving_default"](t)
            pred = list(res.values())[0]
        else:
            pred = m(t, training=False)
        return pred[0].numpy()


# ── OCR ────────────────────────────────────────────────────────────────────────

_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Parámetros de EasyOCR más permisivos que los valores por defecto:
# - text_threshold 0.7→0.4 : detecta texto con menos certeza
# - low_text       0.4→0.3 : umbral más bajo para conectar regiones de texto
# - min_size       10→5    : acepta caracteres más pequeños (en px)
# - contrast_ths   0.1→0.05: aplica adjust_contrast en más imágenes
_OCR_PARAMS = dict(
    detail=1,
    paragraph=False,
    text_threshold=0.4,
    low_text=0.3,
    min_size=5,
    contrast_ths=0.05,
    adjust_contrast=0.5,
)


def _leer_ocr(bgr_placa):
    """
    Lee el texto del recorte probando 6 variantes de preprocesado.
    Retorna (texto, confianza_acumulada) de la variante de mayor confianza.
    """
    ocr = _obtener_ocr()

    gris    = cv2.cvtColor(bgr_placa, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gris_cl = clahe.apply(gris)

    # Sharpening: realza bordes de caracteres desenfocados
    k_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    sharpened = cv2.filter2D(bgr_placa, -1, k_sharp)

    # Invertida: útil para placas con fondo oscuro y texto claro
    invertida = cv2.bitwise_not(bgr_placa)

    # Ecualización por canal: mejora el contraste global
    b, g, r = cv2.split(bgr_placa)
    eq = cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])

    variantes = {
        "color"    : bgr_placa,
        "grises"   : cv2.cvtColor(gris,    cv2.COLOR_GRAY2BGR),
        "clahe"    : cv2.cvtColor(gris_cl, cv2.COLOR_GRAY2BGR),
        "sharpened": sharpened,
        "invertida": invertida,
        "eq"       : eq,
    }

    mejor_texto = ""
    mejor_conf  = 0.0

    for nombre, img in variantes.items():
        resultados  = ocr.readtext(img, **_OCR_PARAMS)
        tokens, acc = [], 0.0
        for (_, txt, conf) in resultados:
            if conf >= 0.01:          # umbral muy bajo: acepta detecciones débiles
                tokens.append(txt)
                acc += conf
        if acc > mejor_conf:
            mejor_conf  = acc
            mejor_texto = " ".join(tokens)

    return mejor_texto, mejor_conf


def _leer_ocr_por_caracter(crops):
    """
    Corre EasyOCR en cada crop de carácter individualmente.
    Retorna lista de (char_predicho, confianza) — un elemento por crop.

    EasyOCR tiene contexto limitado en imágenes pequeñas, por lo que
    se escala cada crop a un mínimo de 32 px de alto y se restringe
    al conjunto de caracteres alfanuméricos (allowlist).
    """
    if not crops:
        return []

    ocr = _obtener_ocr()
    resultados = []

    for crop in crops:
        if crop is None or crop.size == 0:
            resultados.append(("?", 0.0))
            continue

        h, w = crop.shape[:2]
        # Escalar al mínimo para que EasyOCR procese bien caracteres pequeños
        if h < 32:
            factor = max(2, 32 // h)
            crop_proc = cv2.resize(crop, (w * factor, h * factor),
                                   interpolation=cv2.INTER_CUBIC)
        else:
            crop_proc = crop

        try:
            res = ocr.readtext(
                crop_proc, detail=1,
                allowlist=_CHARSET,
                paragraph=False,
            )
            if res:
                best = max(res, key=lambda x: x[2])
                char_text = best[1].strip().upper()
                # Tomar solo el primer carácter alfanumérico
                char_text = next(
                    (c for c in char_text if c in _CHARSET), "?"
                )
                char_conf = float(best[2])
            else:
                char_text, char_conf = "?", 0.0
        except Exception:
            char_text, char_conf = "?", 0.0

        resultados.append((char_text, char_conf))

    return resultados


# ── Visualización del pipeline completo ───────────────────────────────────────

def _construir_collage(imagen_bgr, bbox_px, recorte_bgr,
                       vis_chars, texto, conf_ocr, n_chars, chars_conf=None):
    """
    Genera un collage en 2 filas:
      Fila 1: imagen original  |  imagen con bbox  |  recorte color
      Fila 2: chars segmentados (tira)  |  panel de resultado
    """
    alto_f1 = 260
    alto_f2 = 120
    sep_v   = 4   # separador vertical (px)
    sep_h   = 4   # separador horizontal

    color_sep = (200, 200, 200)

    def _escalar(img, alto):
        h, w = img.shape[:2]
        nw   = max(1, int(w * alto / h))
        return cv2.resize(img, (nw, alto))

    def _titulo(img, txt):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(out, txt, (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return out

    def _sep_v(alto):
        return np.full((alto, sep_v, 3), color_sep, dtype=np.uint8)

    def _sep_h(ancho):
        return np.full((sep_h, ancho, 3), color_sep, dtype=np.uint8)

    # ── Fila 1 ──────────────────────────────────────────────────────────────
    p1 = _titulo(_escalar(imagen_bgr.copy(), alto_f1), "1. Original")

    img_bbox = imagen_bgr.copy()
    if bbox_px:
        x1, y1, x2, y2 = bbox_px
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bbox, "placa", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    p2 = _titulo(_escalar(img_bbox, alto_f1), "2. Deteccion bbox")

    p3 = _titulo(_escalar(recorte_bgr, alto_f1), "3. Crop placa")

    # Igualar alto
    ancho_f1 = p1.shape[1] + sep_v + p2.shape[1] + sep_v + p3.shape[1]
    fila1 = np.hstack([p1, _sep_v(alto_f1), p2, _sep_v(alto_f1), p3])

    # ── Fila 2 ──────────────────────────────────────────────────────────────
    # Panel de caracteres segmentados
    if vis_chars is not None:
        p4 = _titulo(_escalar(vis_chars, alto_f2), f"4. Chars ({n_chars} encontrados)")
    else:
        p4 = np.full((alto_f2, 300, 3), 40, dtype=np.uint8)
        cv2.putText(p4, "Sin segmentacion", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

    # Panel de resultado
    ancho_res = max(300, fila1.shape[1] - p4.shape[1] - sep_v)
    p5 = np.full((alto_f2, ancho_res, 3), 30, dtype=np.uint8)
    linea1 = f"PLACA: {texto if texto else '(no legible)'}"
    linea2 = f"Conf OCR: {conf_ocr:.2f}   Chars: {n_chars}"
    cv2.putText(p5, "5. Resultado", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(p5, linea1, (8, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
    cv2.putText(p5, linea2, (8, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    # Confianza por carácter en una línea compacta
    if chars_conf:
        resumen = "  ".join(
            f"{c}:{cf:.2f}" for c, cf in chars_conf[:10]
        )
        cv2.putText(p5, resumen[:60], (8, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 200, 255), 1)

    # Ajustar ancho de fila 2 para que coincida con fila 1
    objetivo = fila1.shape[1]
    ancho_p4 = p4.shape[1]
    ancho_p5_real = objetivo - ancho_p4 - sep_v

    if ancho_p5_real < 100:
        ancho_p5_real = 100

    p5 = cv2.resize(p5, (ancho_p5_real, alto_f2))
    fila2 = np.hstack([p4, _sep_v(alto_f2), p5])

    # Ajustar anchos si difieren
    if fila1.shape[1] != fila2.shape[1]:
        objetivo = max(fila1.shape[1], fila2.shape[1])
        if fila1.shape[1] < objetivo:
            pad = np.full((alto_f1, objetivo - fila1.shape[1], 3), 200, dtype=np.uint8)
            fila1 = np.hstack([fila1, pad])
        if fila2.shape[1] < objetivo:
            pad = np.full((alto_f2, objetivo - fila2.shape[1], 3), 200, dtype=np.uint8)
            fila2 = np.hstack([fila2, pad])

    collage = np.vstack([fila1, _sep_h(fila1.shape[1]), fila2])
    return collage


# ── API principal ──────────────────────────────────────────────────────────────

def pipeline(ruta_imagen, modelo=MODELO_POR_DEFECTO,
             guardar_en=None, mostrar=False):
    """
    Ejecuta el pipeline ALPR completo.

    Args:
        ruta_imagen : str   — ruta a la imagen del vehículo
        modelo      : str   — "keras" | "pytorch" | "tensorflow"
        guardar_en  : str   — ruta para guardar la imagen anotada (opcional)
        mostrar     : bool  — abrir ventana con el resultado

    Returns:
        dict:
            "placa"           : str
            "confianza"       : float  (confianza OCR acumulada)
            "bbox"            : [x1, y1, x2, y2] en px absolutos, o None
            "n_chars"         : int
            "imagen_resultado": np.ndarray
            "tiempo_ms"       : dict con tiempos de cada etapa
    """
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"Imagen no encontrada: {ruta_imagen}")

    tiempos = {}

    # ── 1. Carga ───────────────────────────────────────────────────────────
    t0  = time.time()
    bgr = cv2.imread(ruta_imagen)
    tiempos["carga_ms"] = (time.time() - t0) * 1000

    # ── 2. Detección de bbox ───────────────────────────────────────────────
    t0 = time.time()
    bbox_norm = _predecir_bbox(bgr, modelo)
    tiempos["deteccion_ms"] = (time.time() - t0) * 1000

    h_img, w_img = bgr.shape[:2]
    x1 = int(bbox_norm[0] * w_img)
    y1 = int(bbox_norm[1] * h_img)
    x2 = int(bbox_norm[2] * w_img)
    y2 = int(bbox_norm[3] * h_img)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    bbox_px = [x1, y1, x2, y2]

    print(f"Bbox predicho: [{x1}, {y1}, {x2}, {y2}]")

    # ── 3. Extracción ROI con margen ──────────────────────────────────────────
    t0 = time.time()

    # Expandir bbox 25% horizontal y 35% vertical (mín. 15 px absolutos).
    # Un bbox ajustado corta bordes de caracteres. Para placas pequeñas
    # (<20 px de alto) el margen es el factor más importante para el OCR.
    ancho_bbox = max(1, x2 - x1)
    alto_bbox  = max(1, y2 - y1)
    mg_x = max(15, int(ancho_bbox * 0.25))
    mg_y = max(10, int(alto_bbox  * 0.35))
    x1m = max(0,     x1 - mg_x)
    y1m = max(0,     y1 - mg_y)
    x2m = min(w_img, x2 + mg_x)
    y2m = min(h_img, y2 + mg_y)

    recorte = bgr[y1m:y2m, x1m:x2m].copy()
    tiempos["extraccion_ms"] = (time.time() - t0) * 1000

    if recorte.size == 0 or recorte.shape[0] < 5 or recorte.shape[1] < 10:
        print("Advertencia: recorte degenerado; usando imagen completa como fallback.")
        recorte = bgr.copy()

    # ── 4. Preprocesamiento ────────────────────────────────────────────────
    t0    = time.time()
    pasos = pipeline_completo(recorte)
    tiempos["preproceso_ms"] = (time.time() - t0) * 1000

    recorte_redim = pasos["redim"]

    # ── 5. Segmentación de caracteres ──────────────────────────────────────
    t0 = time.time()
    crops, bboxes_chars, vis_chars, _ = segmentar(recorte)
    tiempos["segmentacion_ms"] = (time.time() - t0) * 1000

    n_chars = len(crops)
    print(f"Caracteres segmentados: {n_chars}")

    # ── 6. Reconocimiento OCR ──────────────────────────────────────────────
    t0 = time.time()
    texto, conf_ocr = _leer_ocr(recorte_redim)
    # OCR por carácter individual sobre cada crop segmentado
    chars_conf  = _leer_ocr_por_caracter(crops)
    texto_chars = "".join(c for c, _ in chars_conf if c != "?")
    tiempos["ocr_ms"] = (time.time() - t0) * 1000

    tiempos["total_ms"] = sum(tiempos.values())

    print("-" * 45)
    print(f"PLACA LEIDA  : {texto if texto else '(no legible)'}")
    print(f"Por carácter : {texto_chars if texto_chars else '(no legible)'}")
    for idx, (ch, cf) in enumerate(chars_conf, 1):
        print(f"  Char {idx}: '{ch}'  conf={cf:.2f}")
    print(f"Conf. OCR    : {conf_ocr:.2f}")
    print(f"Chars segm.  : {n_chars}")
    print(f"Tiempo total : {tiempos['total_ms']:.0f} ms")

    # ── 7. Collage visual ──────────────────────────────────────────────────
    collage = _construir_collage(
        bgr, bbox_px, recorte_redim,
        vis_chars, texto, conf_ocr, n_chars, chars_conf
    )

    if guardar_en:
        cv2.imwrite(guardar_en, collage)
        print(f"Resultado guardado en: {guardar_en}")

    if mostrar:
        cv2.imshow("Pipeline ALPR", collage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "placa"           : texto,
        "confianza"       : conf_ocr,
        "chars_conf"      : chars_conf,    # [(char, conf), ...] por carácter segmentado
        "texto_chars"     : texto_chars,   # cadena reconstruida desde OCR por carácter
        "bbox"            : bbox_px,
        "n_chars"         : n_chars,
        "imagen_resultado": collage,
        "tiempo_ms"       : tiempos,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ALPR: detección + segmentación + OCR."
    )
    parser.add_argument("--imagen",   required=True,
                        help="Foto del vehículo")
    parser.add_argument("--modelo",   default=MODELO_POR_DEFECTO,
                        choices=["keras", "pytorch", "tensorflow"],
                        help="Modelo CNN a usar (default: keras)")
    parser.add_argument("--guardar",  default="resultado_pipeline.png",
                        help="Guardar imagen anotada (default: resultado_pipeline.png)")
    parser.add_argument("--mostrar",  action="store_true",
                        help="Abrir ventana con el resultado")
    args = parser.parse_args()

    resultado = pipeline(
        ruta_imagen = args.imagen,
        modelo      = args.modelo,
        guardar_en  = args.guardar,
        mostrar     = args.mostrar,
    )

    print("\nResumen:")
    print(f"  placa     : {resultado['placa']}")
    print(f"  confianza : {resultado['confianza']:.2f}")
    print(f"  bbox      : {resultado['bbox']}")
    print(f"  n_chars   : {resultado['n_chars']}")
    print(f"  tiempos   : {resultado['tiempo_ms']}")


if __name__ == "__main__":
    main()
