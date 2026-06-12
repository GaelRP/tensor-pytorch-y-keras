"""
train_tensorflow.py
-------------------
Entrena una CNN con TensorFlow puro para detectar la región
de la placa en fotos de carros completos.

A diferencia de train_keras.py que usa la API de alto nivel (keras),
aquí usamos tf.Module y GradientTape para construir y entrenar
el modelo manualmente, mostrando cómo funciona TensorFlow por dentro.

La CNN aprende a predecir 4 valores: [x_min, y_min, x_max, y_max]
todos normalizados entre 0 y 1.

Conceptos clave implementados aquí:
  - CONVOLUCIÓN  : tf.keras.layers.Conv2D (pero entrenado manualmente)
  - RELU         : tf.nn.relu()
  - POOLING      : tf.keras.layers.MaxPool2D
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf

from utils import cargar_dataset


# ── Rutas del dataset ────────────────────────────────────────────────────────

CARPETA_IMAGENES    = "dataset_placas/images"
CARPETA_ANOTACIONES = "dataset_placas/annotations"
RUTA_MODELO         = "models/modelo_tensorflow"

# ── Hiperparámetros ──────────────────────────────────────────────────────────

EPOCAS           = 30
BATCH_SIZE       = 16
TASA_APRENDIZAJE = 0.001


class CNNDetectorPlacasTF(tf.Module):
    """
    Modelo definido como tf.Module en lugar de keras.Model.
    Esto obliga a definir cada capa manualmente y a hacer
    el paso forward explícitamente, igual que en PyTorch.

    ¿Qué hace cada tipo de capa?

    CONVOLUCIÓN (tf.keras.layers.Conv2D):
        Un kernel (matriz de pesos) se desliza sobre la imagen.
        En cada posición multiplica elemento a elemento la región
        de la imagen con el kernel y suma los resultados.
        Resultado: un mapa de activación que resalta dónde
        aparece el patrón que aprendió ese filtro.
        Ejemplo visual con filtro detector de bordes verticales:
          imagen  kernel     resultado
          1 0 0   -1 0 1      borde!
          1 0 0   -1 0 1  →   borde!
          1 0 0   -1 0 1      borde!

    RELU (tf.nn.relu):
        Función de activación no lineal.
        f(x) = x  si x > 0
        f(x) = 0  si x ≤ 0
        Sin ReLU, sin importar cuántas capas tenga la red,
        todo colapsaría a una sola transformación lineal
        y no podría aprender funciones complejas como
        "la placa está en la esquina inferior derecha".

    POOLING (tf.keras.layers.MaxPool2D):
        Toma ventanas de 2x2 y guarda solo el máximo.
        Antes del pool: 224x224 → Después: 112x112
        Hace la red invariante a pequeñas traslaciones:
        si la placa se mueve 1 pixel, el max pooling
        puede seguir detectándola igual de bien.
    """

    def __init__(self):
        super(CNNDetectorPlacasTF, self).__init__()

        # ── Bloque 1 ──────────────────────────────────────────────────────────
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same"
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        # ── Bloque 2 ──────────────────────────────────────────────────────────
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same"
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        # ── Bloque 3 ──────────────────────────────────────────────────────────
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same"
        )
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        # ── Cabeza de regresión ───────────────────────────────────────────────
        self.flatten = tf.keras.layers.Flatten()
        self.dense1  = tf.keras.layers.Dense(units=128)
        self.dropout = tf.keras.layers.Dropout(rate=0.3)
        self.dense2  = tf.keras.layers.Dense(units=4)

    def __call__(self, x, entrenando=False):
        """
        Flujo de datos a través de la red.
        El parámetro 'entrenando' activa/desactiva el Dropout.
        """

        # Bloque 1: Conv → ReLU → Pool
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        # Bloque 2: Conv → ReLU → Pool
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.pool2(x)

        # Bloque 3: Conv → ReLU → Pool
        x = self.conv3(x)
        x = tf.nn.relu(x)
        x = self.pool3(x)

        # Aplanar y capas densas
        x = self.flatten(x)

        x = self.dense1(x)
        x = tf.nn.relu(x)

        x = self.dropout(x, training=entrenando)

        x = self.dense2(x)

        # Sigmoid para obtener salida en [0, 1]
        x = tf.sigmoid(x)

        return x


def calcular_perdida(modelo, imagenes_batch, etiquetas_batch, entrenando):
    """
    Calcula el error MSE entre predicciones y etiquetas reales.
    Se separa en función propia para poder llamarla también
    durante la validación sin gradientes.
    """
    predicciones = modelo(imagenes_batch, entrenando=entrenando)
    perdida      = tf.reduce_mean(tf.square(predicciones - etiquetas_batch))

    return perdida


def paso_entrenamiento(modelo, optimizador, imagenes_batch, etiquetas_batch):
    """
    Un paso de entrenamiento completo:
    1. GradientTape registra todas las operaciones
    2. Se calcula la pérdida
    3. Se calculan los gradientes automáticamente
    4. El optimizador actualiza los pesos

    GradientTape es lo que diferencia a TensorFlow puro de Keras:
    tenemos control total sobre cuándo y cómo se calculan los gradientes.
    """
    with tf.GradientTape() as tape:
        perdida = calcular_perdida(
            modelo,
            imagenes_batch,
            etiquetas_batch,
            entrenando=True
        )

    # Calcular gradiente de la pérdida respecto a cada parámetro del modelo
    gradientes = tape.gradient(perdida, modelo.trainable_variables)

    # Aplicar gradientes: actualizar los pesos
    optimizador.apply_gradients(zip(gradientes, modelo.trainable_variables))

    return perdida


def crear_dataset_tf(imagenes, etiquetas):
    """
    Convierte arrays numpy a tf.data.Dataset, que es la forma
    eficiente de alimentar datos en TensorFlow.
    Hace shuffle (mezcla) y divide en batches automáticamente.
    """
    dataset = tf.data.Dataset.from_tensor_slices((imagenes, etiquetas))
    dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def entrenar():

    if not os.path.exists(CARPETA_IMAGENES):
        print(f"ERROR: No se encontró la carpeta '{CARPETA_IMAGENES}'")
        return

    os.makedirs("models", exist_ok=True)

    print("Cargando dataset...")
    imagenes, etiquetas = cargar_dataset(CARPETA_IMAGENES, CARPETA_ANOTACIONES)

    X_train, X_val, y_train, y_val = train_test_split(
        imagenes,
        etiquetas,
        test_size=0.2,
        random_state=42
    )

    print(f"Entrenamiento: {len(X_train)} imágenes")
    print(f"Validación   : {len(X_val)} imágenes")

    # Crear datasets de TensorFlow
    dataset_train = crear_dataset_tf(X_train, y_train)
    dataset_val   = crear_dataset_tf(X_val,   y_val)

    print("\nConstruyendo modelo TensorFlow...")
    modelo      = CNNDetectorPlacasTF()
    optimizador = tf.optimizers.Adam(learning_rate=TASA_APRENDIZAJE)

    # Inicializar las variables del modelo pasando un batch de prueba
    batch_prueba = tf.zeros((1, 224, 224, 3))
    modelo(batch_prueba, entrenando=False)
    print(f"Parámetros totales: {sum(v.numpy().size for v in modelo.trainable_variables):,}")

    print(f"\nEntrenando por {EPOCAS} épocas...")

    for epoca in range(1, EPOCAS + 1):

        # ── Fase de entrenamiento ─────────────────────────────────────────────
        perdidas_train = []

        for imagenes_batch, etiquetas_batch in dataset_train:
            perdida = paso_entrenamiento(modelo, optimizador, imagenes_batch, etiquetas_batch)
            perdidas_train.append(perdida.numpy())

        perdida_promedio_train = np.mean(perdidas_train)

        # ── Fase de validación ────────────────────────────────────────────────
        perdidas_val = []

        for imagenes_batch, etiquetas_batch in dataset_val:
            perdida = calcular_perdida(
                modelo,
                imagenes_batch,
                etiquetas_batch,
                entrenando=False
            )
            perdidas_val.append(perdida.numpy())

        perdida_promedio_val = np.mean(perdidas_val)

        print(f"Época {epoca:3d}/{EPOCAS} | "
              f"Loss train: {perdida_promedio_train:.4f} | "
              f"Loss val: {perdida_promedio_val:.4f}")

    # Guardar el modelo en formato SavedModel de TensorFlow.
    # Se define la firma explícita para que detector_placas.py
    # pueda encontrarla con modelo.signatures['serving_default'].
    # Sin esto, tf.saved_model.load() devuelve un _UserObject
    # sin firmas accesibles y no se puede llamar directamente.
    funcion_inferencia = tf.function(
        lambda x: modelo(x, entrenando=False),
        input_signature=[
            tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)
        ]
    )

    tf.saved_model.save(
        modelo,
        RUTA_MODELO,
        signatures={'serving_default': funcion_inferencia}
    )
    print(f"\nModelo guardado en: {RUTA_MODELO}/")


if __name__ == "__main__":
    entrenar()