import argparse
import os
import random

import numpy as np
import tensorflow as tf

from cnn_trainer import MODELOS

IMG_SIZE = (224, 224)  # el mismo tamaño con el que se entrenó
EXPERIMENTO = "experimento_1"
CLASES = [str(i) for i in range(10)]  # carpetas del dataset: 0..9 (orden alfabético = índice)


def cargar_modelo(nombre, experimento=EXPERIMENTO, output_dir="resultados"):
    ruta = os.path.join(output_dir, experimento, nombre, f"{nombre}.keras")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el modelo: {ruta}")
    return tf.keras.models.load_model(ruta)


def predecir(modelo, nombre, ruta_imagen):
    """Devuelve (clase_predicha, probabilidad, vector_de_probabilidades)."""
    # Mismo preprocesamiento que en el entrenamiento: decode -> resize -> preprocess_input del modelo
    img = tf.io.read_file(ruta_imagen)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = MODELOS[nombre][1](img)
    probs = modelo.predict(tf.expand_dims(img, 0), verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASES[idx], float(probs[idx]), probs


def imagen_aleatoria(dataset_dir="dataset"):
    """Elige una imagen cualquiera del dataset y devuelve (ruta, clase_real)."""
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    rutas = []
    for clase in CLASES:
        for raiz, _, archivos in os.walk(os.path.join(dataset_dir, clase)):
            rutas += [(os.path.join(raiz, a), clase) for a in archivos
                      if os.path.splitext(a)[1].lower() in extensiones]
    return random.choice(rutas)


def main():
    parser = argparse.ArgumentParser(description="Clasifica un dígito en una imagen con un modelo entrenado.")
    parser.add_argument("imagen", nargs="?",
                        help="Ruta de la imagen a clasificar (si se omite, se elige una al azar del dataset)")
    parser.add_argument("--modelo", default="MobileNet", choices=list(MODELOS),
                        help="Modelo entrenado a usar (por defecto MobileNet)")
    parser.add_argument("--experimento", default=EXPERIMENTO)
    parser.add_argument("--top", type=int, default=3, help="Cuántas clases más probables mostrar")
    args = parser.parse_args()

    real = None
    if args.imagen is None:
        args.imagen, real = imagen_aleatoria()
    elif not os.path.exists(args.imagen):
        parser.error(f"No existe la imagen: {args.imagen}")
    print(f"Imagen: {args.imagen}")

    modelo = cargar_modelo(args.modelo, args.experimento)
    clase, prob, probs = predecir(modelo, args.modelo, args.imagen)

    if real is not None:
        print(f"Número real (carpeta del dataset): {real}")
    print(f"\nCon una probabilidad de {prob * 100:.2f}% en la imagen se ve el número {clase}")

    if args.top > 1:
        print(f"\nTop {args.top}:")
        for i in np.argsort(probs)[::-1][:args.top]:
            print(f"  {CLASES[i]}: {probs[i] * 100:.2f}%")


if __name__ == "__main__":
    main()
