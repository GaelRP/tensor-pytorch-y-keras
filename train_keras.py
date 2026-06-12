"""
train_keras.py
--------------
Entrena una CNN con Keras (backend TensorFlow) para detectar
la región de la placa en fotos de carros completos.

La CNN aprende a predecir 4 valores: [x_min, y_min, x_max, y_max]
todos normalizados entre 0 y 1.

Conceptos clave implementados aquí:
  - CONVOLUCIÓN  : capas Conv2D
  - RELU         : función de activación en cada capa
  - POOLING      : capas MaxPooling2D
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

# Desde TensorFlow 2.16, Keras es un paquete separado llamado "keras"
# Si tienes TF 2.15 o menor, usa: from tensorflow import keras

from tensorflow import keras
from tensorflow.keras import layers
from utils import cargar_dataset


# ── Rutas del dataset ────────────────────────────────────────────────────────

CARPETA_IMAGENES     = "dataset_placas/images"
CARPETA_ANOTACIONES  = "dataset_placas/annotations"
RUTA_MODELO          = "models/modelo_keras.keras"

# ── Hiperparámetros ──────────────────────────────────────────────────────────

EPOCAS       = 30
BATCH_SIZE   = 16
TASA_APRENDIZAJE = 0.001


def construir_modelo():
    """
    Construye la arquitectura de la CNN.

    ¿Qué hace cada tipo de capa?

    CONVOLUCIÓN (Conv2D):
        Aplica un filtro (kernel) que recorre la imagen pixel por pixel.
        Cada filtro aprende a detectar un patrón: bordes, esquinas, texturas.
        Ejemplo con kernel 3x3: toma 9 píxeles, los multiplica por 9 pesos
        aprendibles y los suma en un solo valor. Esto se repite en toda la imagen.
        El resultado es un "mapa de características" que resalta dónde aparece
        ese patrón. Con 32 filtros obtenemos 32 mapas distintos.

    RELU (Rectified Linear Unit):
        Es la función de activación aplicada después de cada convolución.
        Fórmula: f(x) = max(0, x)
        Si el valor es negativo → se vuelve 0 (se apaga la neurona).
        Si el valor es positivo → se deja igual.
        ¿Por qué? Introduce no-linealidad sin dificultar el entrenamiento.
        Sin activaciones, la red solo podría aprender combinaciones lineales
        (rectas), lo cual es insuficiente para detectar placas en posiciones
        variables.

    POOLING (MaxPooling2D):
        Reduce el tamaño espacial de los mapas de características.
        Con pool_size=(2,2): toma ventanas de 2x2 píxeles y conserva
        solo el valor máximo. Resultado: la imagen se reduce a la mitad.
        ¿Por qué? Dos razones:
          1. Reduce la cantidad de parámetros → más rápido de entrenar.
          2. Hace la detección más robusta a pequeños desplazamientos.
             Si la placa se mueve 1 pixel, el pooling "absorbe" ese cambio.
    """

    entrada = keras.Input(shape=(224, 224, 3))

    # Conv2D: 32 filtros de 3x3, activación ReLU
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(entrada)
    # MaxPooling: reduce de 224x224 a 112x112
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(x)
    # Reduce de 112x112 a 56x56
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")(x)
    # Reduce de 56x56 a 28x28
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # Aplanar el volumen 3D a un vector 1D
    x = layers.Flatten()(x)
    # Capa densa para combinar todas las características
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(rate=0.3)(x)
    salida = layers.Dense(units=4, activation="sigmoid")(x)
    modelo = keras.Model(inputs=entrada, outputs=salida)
    return modelo


def entrenar():

    # Verificar que existan las carpetas
    if not os.path.exists(CARPETA_IMAGENES):
        print(f"ERROR: No se encontró la carpeta '{CARPETA_IMAGENES}'")
        return

    # Crear la carpeta de modelos si no existe
    os.makedirs("models", exist_ok=True)

    print("Cargando dataset...")
    imagenes, etiquetas = cargar_dataset(CARPETA_IMAGENES, CARPETA_ANOTACIONES)

    # Dividir en entrenamiento (80%) y validación (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        imagenes,
        etiquetas,
        test_size=0.2,
        random_state=42
    )

    print(f"Entrenamiento: {len(X_train)} imágenes")
    print(f"Validación   : {len(X_val)} imágenes")

    print("\nConstruyendo modelo Keras...")
    modelo = construir_modelo()
    modelo.summary()

    # Compilar: loss MSE mide qué tan lejos está el bounding box predicho
    # del bounding box real. Adam es el optimizador más usado por ser estable.
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TASA_APRENDIZAJE),
        loss="mean_squared_error",
        metrics=["mae"]
    )

    print(f"\nEntrenando por {EPOCAS} épocas...")
    historial = modelo.fit(
        X_train,
        y_train,
        epochs=EPOCAS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Guardar el modelo entrenado
    modelo.save(RUTA_MODELO)
    print(f"\nModelo guardado en: {RUTA_MODELO}")

    # Mostrar pérdida final
    perdida_final    = historial.history["loss"][-1]
    perdida_val_final = historial.history["val_loss"][-1]
    print(f"Loss entrenamiento : {perdida_final:.4f}")
    print(f"Loss validación    : {perdida_val_final:.4f}")


if __name__ == "__main__":
    entrenar()