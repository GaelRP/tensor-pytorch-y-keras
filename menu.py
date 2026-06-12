"""
menu.py
-------
Menu interactivo en terminal para detectar placas.

Uso:
    python menu.py
"""

import os
import sys
import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def limpiar_pantalla():
    os.system("cls" if os.name == "nt" else "clear")


def imprimir_encabezado():
    
    print("   DETECTOR DE PLACAS VEHICULARES")
    
    print()


def elegir_modelo():
    modelos = [
        ("keras",      "models/modelo_keras.keras"),
        ("pytorch",    "models/modelo_pytorch.pth"),
        ("tensorflow", "models/modelo_tensorflow"),
    ]

    print("Modelo a usar:\n")

    for i, (nombre, ruta) in enumerate(modelos, start=1):
        estado = "listo" if os.path.exists(ruta) else "no entrenado"
        print(f"  [{i}] {nombre:<14} ({estado})")

    print("  [0] Salir")
    print()

    while True:
        opcion = input("Opcion: ").strip()

        if opcion == "0":
            print("\nHasta luego.\n")
            sys.exit(0)

        if opcion in ("1", "2", "3"):
            indice = int(opcion) - 1
            nombre, ruta = modelos[indice]

            if not os.path.exists(ruta):
                print(f"\nEl modelo '{nombre}' no esta entrenado.")
                print(f"Ejecuta primero: python train_{nombre}.py\n")
                continue

            return nombre

        print("Opcion no valida.\n")


def elegir_imagen():
    print()
    print("Como quieres seleccionar la imagen:\n")
    print("  [1] Escribir la ruta manualmente")
    print("  [2] Elegir de esta carpeta")
    print("  [3] Elegir del dataset")
    print("  [0] Volver")
    print()

    while True:
        opcion = input("Opcion: ").strip()

        if opcion == "0":
            return None

        if opcion == "1":
            return pedir_ruta_manual()

        if opcion == "2":
            return elegir_de_carpeta(".", ["*.png", "*.jpg", "*.jpeg"])

        if opcion == "3":
            return elegir_de_carpeta(
                "dataset_placas/images",
                ["*.png", "*.jpg", "*.jpeg"]
            )

        print("Opcion no valida.\n")


def pedir_ruta_manual():
    print()

    while True:
        ruta = input("Ruta de la imagen: ").strip().strip('"')

        if not ruta:
            print("Escribe una ruta valida.")
            continue

        if not os.path.exists(ruta):
            print(f"Archivo no encontrado: {ruta}")
            otra = input("Intentar con otra ruta? (s/n): ").strip().lower()
            if otra != "s":
                return None
            continue

        return ruta


def elegir_de_carpeta(carpeta, patrones):
    if not os.path.exists(carpeta):
        print(f"\nCarpeta no encontrada: {carpeta}")
        return None

    imagenes = []
    for patron in patrones:
        imagenes.extend(glob.glob(os.path.join(carpeta, patron)))

    imagenes = sorted(imagenes)

    if not imagenes:
        print(f"\nNo se encontraron imagenes en: {carpeta}")
        return None

    print(f"\nImagenes en '{carpeta}':\n")

    limite   = 20
    mostradas = imagenes[:limite]

    for i, ruta_img in enumerate(mostradas, start=1):
        nombre = os.path.basename(ruta_img)
        print(f"  [{i:2d}] {nombre}")

    if len(imagenes) > limite:
        print(f"\n  ... y {len(imagenes) - limite} imagenes mas.")

    print("  [ 0] Volver")
    print()

    while True:
        opcion = input("Numero de imagen: ").strip()

        if opcion == "0":
            return None

        if opcion.isdigit():
            indice = int(opcion) - 1
            if 0 <= indice < len(mostradas):
                return mostradas[indice]

        print(f"Numero no valido. Elige entre 1 y {len(mostradas)}.")


def confirmar_y_ejecutar(modelo, ruta_imagen):
    print()
    
    print(f"Modelo : {modelo}")
    print(f"Imagen : {ruta_imagen}")
    
    print()

    confirmar = input("Ejecutar? (s/n): ").strip().lower()

    if confirmar != "s":
        print("\nCancelado.\n")
        return

    print()
    comando = f'python detector_placas.py --imagen "{ruta_imagen}" --modelo {modelo}'
    os.system(comando)

    print()
    input("Presiona Enter para volver al menu...")


def menu_principal():
    while True:
        limpiar_pantalla()
        imprimir_encabezado()

        modelo = elegir_modelo()

        print()
        print("-" * 40)

        ruta_imagen = elegir_imagen()

        if ruta_imagen is None:
            continue

        confirmar_y_ejecutar(modelo, ruta_imagen)


if __name__ == "__main__":
    menu_principal()