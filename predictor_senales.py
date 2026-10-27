import json
import os

import cv2
import numpy as np
import tensorflow as tf

from modelos_cnn import CLASES, IMG_SIZE, MODELOS


class PredictorSenales:
    """
    Carga los pesos (.keras) que dejó el entrenamiento y predice la señal
    de una foto. Por defecto usa el mejor modelo de seleccion.json (el de
    menor val_loss); se puede forzar otro con `modelo="VGG16"`, etc.

        predictor = PredictorSenales("reto2_exp1")
        clase, prob = predictor.predecir("foto.jpg")   # o un frame de OpenCV (BGR)
    """

    def __init__(self, experimento="reto2_exp1", modelo=None, output_dir="resultados"):
        carpeta = os.path.join(output_dir, experimento)
        with open(os.path.join(carpeta, "seleccion.json"), encoding="utf-8") as f:
            seleccion = json.load(f)

        self.nombre = modelo or seleccion["mejores"][0]["modelo"]
        self.img_size = tuple(seleccion.get("img_size", IMG_SIZE))
        # Las carpetas del dataset son 0..3: el índice indica la señal
        self.clases = CLASES
        self._preprocesar = MODELOS[self.nombre][1]
        self.modelo = tf.keras.models.load_model(
            os.path.join(carpeta, self.nombre, f"{self.nombre}.keras"))
        self.modelo(np.zeros((1,) + self.img_size + (3,), np.float32))  # calentamiento

    @staticmethod
    def _a_proporcion(foto):
        """
        Las fotos de entrenamiento son 3:4 (vertical) o 4:3 (apaisada) y se
        aplastan a 224x224. Un frame 16:9 o 9:16 se rellena con bordes hasta
        esa proporción (sin recortar, para no perder señales en las orillas),
        así las formas se deforman igual que en el entrenamiento.
        """
        alto, ancho = foto.shape[:2]
        proporcion = 3 / 4 if alto >= ancho else 4 / 3  # ancho / alto
        if ancho / alto < proporcion:  # muy angosto: se agregan lados
            extra = (int(alto * proporcion) - ancho) // 2
            return cv2.copyMakeBorder(foto, 0, 0, extra, extra, cv2.BORDER_REPLICATE)
        extra = (int(ancho / proporcion) - alto) // 2  # muy ancho: arriba y abajo
        return cv2.copyMakeBorder(foto, extra, extra, 0, 0, cv2.BORDER_REPLICATE)

    def predecir(self, foto):
        """Devuelve (clase, probabilidad). `foto` es una ruta o un frame BGR."""
        if isinstance(foto, str):
            # Sin rotar por EXIF, igual que tf.io en el entrenamiento
            foto = cv2.imread(foto, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            if foto is None:
                raise FileNotFoundError("No se pudo leer la imagen.")
        foto = self._a_proporcion(foto)
        rgb = cv2.cvtColor(foto, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.img_size[::-1]).astype(np.float32)
        probs = self.modelo(self._preprocesar(rgb[None]), training=False).numpy()[0]
        i = int(np.argmax(probs))
        self.ultimas_probs = probs  # las 4 probabilidades, para depurar en pantalla
        return self.clases[i], float(probs[i])


if __name__ == "__main__":
    import sys
    experimento = sys.argv[2] if len(sys.argv) > 2 else "reto2_exp1"
    p = PredictorSenales(experimento)
    clase, prob = p.predecir(sys.argv[1])
    print(f"[{p.nombre}] {clase} ({prob * 100:.1f}%)")
