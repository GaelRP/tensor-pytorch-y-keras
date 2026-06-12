"""
train_pytorch.py
----------------
Entrena una CNN con PyTorch para detectar la región de la placa
en fotos de carros completos.

La CNN aprende a predecir 4 valores: [x_min, y_min, x_max, y_max]
todos normalizados entre 0 y 1.

Conceptos clave implementados aquí:
  - CONVOLUCIÓN  : nn.Conv2d
  - RELU         : F.relu()
  - POOLING      : nn.MaxPool2d
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import cargar_dataset


# ── Rutas del dataset ────────────────────────────────────────────────────────

CARPETA_IMAGENES    = "dataset_placas/images"
CARPETA_ANOTACIONES = "dataset_placas/annotations"
RUTA_MODELO         = "models/modelo_pytorch.pth"

# ── Hiperparámetros ──────────────────────────────────────────────────────────

EPOCAS           = 30
BATCH_SIZE       = 16
TASA_APRENDIZAJE = 0.001


class DatasetPlacas(Dataset):
    """
    Clase que envuelve nuestros arrays de numpy en un formato
    que PyTorch entiende para poder usar DataLoader.

    PyTorch espera tensores con el canal primero: (C, H, W)
    OpenCV y numpy usan (H, W, C), entonces hay que reordenar.
    """

    def __init__(self, imagenes, etiquetas):
        # Cambiar de (N, H, W, C) a (N, C, H, W) para PyTorch
        imagenes_transpuestas = imagenes.transpose(0, 3, 1, 2)

        self.imagenes  = torch.tensor(imagenes_transpuestas, dtype=torch.float32)
        self.etiquetas = torch.tensor(etiquetas,             dtype=torch.float32)

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, indice):
        return self.imagenes[indice], self.etiquetas[indice]


class CNNDetectorPlacas(nn.Module):
    """
    Arquitectura de la CNN en PyTorch.

    En PyTorch se define la arquitectura en __init__ y el flujo
    de datos en forward(). Esto da más control que Keras.

    ¿Qué hace cada tipo de capa?

    CONVOLUCIÓN (nn.Conv2d):
        Aplica un filtro (kernel) que recorre la imagen pixel por pixel.
        Cada filtro aprende a detectar un patrón: bordes, esquinas, texturas.
        Con in_channels=3 (RGB), out_channels=32, kernel_size=3:
        toma una región de 3x3x3 (alto x ancho x canales), la multiplica
        por los pesos del filtro y suma todo en un solo número.
        Con padding=1 la imagen mantiene su tamaño espacial.

    RELU (F.relu o nn.ReLU):
        Función de activación: f(x) = max(0, x)
        Valores negativos se vuelven 0, positivos se conservan.
        Sin esto, apilar capas convolucionales sería equivalente a una
        sola capa lineal, porque la composición de funciones lineales
        sigue siendo lineal.

    POOLING (nn.MaxPool2d):
        Ventana de 2x2 que recorre el mapa de características
        conservando solo el valor máximo de cada ventana.
        Reduce el tamaño a la mitad: 224→112→56→28.
        Beneficio: menos parámetros y más robustez ante traslaciones.
    """

    def __init__(self):
        super(CNNDetectorPlacas, self).__init__()

        # ── Capas convolucionales ─────────────────────────────────────────────

        # Bloque 1: 3 canales entrada, 32 filtros salida, kernel 3x3
        self.conv1   = nn.Conv2d(in_channels=3,   out_channels=32,  kernel_size=3, padding=1)
        self.pool1   = nn.MaxPool2d(kernel_size=2, stride=2)
        # Salida: (32, 112, 112)

        # Bloque 2
        self.conv2   = nn.Conv2d(in_channels=32,  out_channels=64,  kernel_size=3, padding=1)
        self.pool2   = nn.MaxPool2d(kernel_size=2, stride=2)
        # Salida: (64, 56, 56)

        # Bloque 3
        self.conv3   = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, padding=1)
        self.pool3   = nn.MaxPool2d(kernel_size=2, stride=2)
        # Salida: (128, 28, 28)

        # ── Capas densas (regresión) ──────────────────────────────────────────

        # 128 * 28 * 28 = 100352 neuronas después de aplanar
        self.fc1     = nn.Linear(128 * 28 * 28, 128)
        self.dropout = nn.Dropout(p=0.3)

        # 4 salidas: [x_min, y_min, x_max, y_max]
        self.fc2     = nn.Linear(128, 4)

    def forward(self, x):
        """
        Define cómo fluyen los datos a través de la red.
        En Keras esto es automático; en PyTorch hay que escribirlo
        explícitamente, lo que permite más flexibilidad.
        """

        # Bloque 1: Conv → ReLU → Pool
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # Bloque 2: Conv → ReLU → Pool
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # Bloque 3: Conv → ReLU → Pool
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        # Aplanar: (batch, 128, 28, 28) → (batch, 100352)
        x = x.reshape(x.size(0), -1)

        # Capa densa con ReLU
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Capa de salida con sigmoid para obtener valores en [0, 1]
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


def entrenar():

    if not os.path.exists(CARPETA_IMAGENES):
        print(f"ERROR: No se encontró la carpeta '{CARPETA_IMAGENES}'")
        return

    os.makedirs("models", exist_ok=True)

    # Usar GPU si está disponible, si no usar CPU
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {dispositivo}")

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

    # Crear Datasets y DataLoaders de PyTorch
    dataset_train = DatasetPlacas(X_train, y_train)
    dataset_val   = DatasetPlacas(X_val,   y_val)

    loader_train  = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_val    = DataLoader(dataset_val,   batch_size=BATCH_SIZE, shuffle=False)

    print("\nConstruyendo modelo PyTorch...")
    modelo = CNNDetectorPlacas()
    modelo = modelo.to(dispositivo)
    print(modelo)

    # MSE como función de pérdida para regresión de bounding box
    criterio   = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)

    print(f"\nEntrenando por {EPOCAS} épocas...")

    for epoca in range(1, EPOCAS + 1):

        # ── Fase de entrenamiento ─────────────────────────────────────────────
        modelo.train()
        perdida_total_train = 0.0

        for imagenes_batch, etiquetas_batch in loader_train:

            imagenes_batch  = imagenes_batch.to(dispositivo)
            etiquetas_batch = etiquetas_batch.to(dispositivo)

            # Borrar gradientes del paso anterior
            optimizador.zero_grad()

            # Paso hacia adelante (forward pass)
            predicciones = modelo(imagenes_batch)

            # Calcular la pérdida
            perdida = criterio(predicciones, etiquetas_batch)

            # Paso hacia atrás (backward pass): calcular gradientes
            perdida.backward()

            # Actualizar pesos con los gradientes calculados
            optimizador.step()

            perdida_total_train += perdida.item()

        perdida_promedio_train = perdida_total_train / len(loader_train)

        # ── Fase de validación ────────────────────────────────────────────────
        modelo.eval()
        perdida_total_val = 0.0

        # torch.no_grad() desactiva el cálculo de gradientes para ahorrar memoria
        with torch.no_grad():
            for imagenes_batch, etiquetas_batch in loader_val:

                imagenes_batch  = imagenes_batch.to(dispositivo)
                etiquetas_batch = etiquetas_batch.to(dispositivo)

                predicciones = modelo(imagenes_batch)
                perdida      = criterio(predicciones, etiquetas_batch)

                perdida_total_val += perdida.item()

        perdida_promedio_val = perdida_total_val / len(loader_val)

        print(f"Época {epoca:3d}/{EPOCAS} | "
              f"Loss train: {perdida_promedio_train:.4f} | "
              f"Loss val: {perdida_promedio_val:.4f}")

    # Guardar solo los pesos del modelo (forma recomendada en PyTorch)
    torch.save(modelo.state_dict(), RUTA_MODELO)
    print(f"\nModelo guardado en: {RUTA_MODELO}")


if __name__ == "__main__":
    entrenar()