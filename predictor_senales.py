import json
import os

import cv2
import numpy as np
import tensorflow as tf

import config_robot
from localizador_color import candidatos, recorte_cuadrado
from modelos_cnn import CLASES, IMG_SIZE, MODELOS


class PredictorSenales:
    """
    Carga los pesos (.keras) que dejó el entrenamiento y predice la señal
    de una foto. Por defecto usa el mejor modelo de seleccion.json (el de
    menor val_loss); se puede forzar otro con `modelo="VGG16"`, etc.

    Dos modos (config_robot.CNN_MODO, o `modo=`):
      "recortes"  localiza la señal por color (localizador_color.py), recorta
                  cada candidato y clasifica SOLO el recorte, así la red no ve
                  el robot ni la pista. Si no hay ningún candidato y
                  CNN_RESPALDO_COMPLETO es True, clasifica el frame completo
                  (el modelo se entrena con las dos cosas).
      "completo"  clasifica siempre el frame completo.

        predictor = PredictorSenales()
        clase, prob = predictor.predecir("foto.jpg")   # o un frame de OpenCV (BGR)
    """

    def __init__(self, modelo=None, carpeta=None, modo=None):
        self.modo = modo or config_robot.CNN_MODO
        self.ultimas_cajas = []  # [(caja, clase, prob)] de la última consulta, para dibujar
        carpeta, seleccion = buscar_modelo(modelo, carpeta)
        self.carpeta = carpeta
        self.nombre = modelo or seleccion["mejores"][0]["modelo"]
        self.img_size = tuple(seleccion.get("img_size", IMG_SIZE))
        self._preprocesar = MODELOS[self.nombre][1]
        self.modelo = tf.keras.models.load_model(
            os.path.join(carpeta, self.nombre, f"{self.nombre}.keras"))
        # El índice de la salida es el número de carpeta del dataset (0=PARE ...).
        # Un modelo de 4 salidas no tiene SIN_SENAL; uno de 5 sí.
        self.clases = CLASES[:self.modelo.output_shape[-1]]
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
        probs = self.probabilidades(foto)
        clase = max(probs, key=probs.get)
        return clase, probs[clase]

    def probabilidades(self, foto):
        """Devuelve {clase: probabilidad}. `foto` es una ruta o un frame BGR."""
        if isinstance(foto, str):
            # Sin rotar por EXIF, igual que tf.io en el entrenamiento
            foto = cv2.imread(foto, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            if foto is None:
                raise FileNotFoundError("No se pudo leer la imagen.")
        if self.modo == "recortes":
            return self._probabilidades_recortes(foto)
        self.ultimas_cajas = []
        return self._probabilidades_completa(foto)

    def _predecir_lote(self, imagenes_bgr):
        """Clasifica varias imágenes BGR ya a tamaño de entrada en UNA sola llamada."""
        lote = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imagenes_bgr]).astype(np.float32)
        return self.modelo(self._preprocesar(lote), training=False).numpy()

    def _probabilidades_completa(self, foto):
        if config_robot.CNN_RELLENO_3_4:
            foto = self._a_proporcion(foto)
        pequena = cv2.resize(foto, self.img_size[::-1], interpolation=cv2.INTER_AREA)
        probs = self._predecir_lote([pequena])[0]
        self.ultimas_probs = probs  # todas las probabilidades, para depurar en pantalla
        return {c: float(p) for c, p in zip(self.clases, probs)}

    def _probabilidades_recortes(self, frame):
        # Igual que al armar el dataset: primero colores estrictos y, si no hay nada,
        # la pasada permisiva (señales pálidas)
        cands = (candidatos(frame) or candidatos(frame, relajado=True))[:config_robot.CNN_MAX_RECORTES]
        self.ultimas_cajas = []
        if not cands:
            if config_robot.CNN_RESPALDO_COMPLETO:
                probs = self._probabilidades_completa(frame)
                # El frame completo se equivoca más (ve el robot y la pista): solo se
                # acepta una señal si está MUY segura; si no, se da por vacía.
                mejor = max((c for c in probs if c != "SIN_SENAL"), key=probs.get, default=None)
                if mejor is None or probs[mejor] >= config_robot.CNN_RESPALDO_MIN_PROB:
                    return probs
            # Nada con color de señal a la vista: no hay señal
            return {c: float(c == "SIN_SENAL") for c in self.clases}

        recortes = [recorte_cuadrado(frame, c["caja"], lado_salida=self.img_size[0]) for c in cands]
        probs = self._predecir_lote(recortes)
        self.ultimas_probs = probs[0]
        self.ultimas_cajas = [(c["caja"], self.clases[int(np.argmax(p))], float(np.max(p)))
                              for c, p in zip(cands, probs)]

        # Gana el candidato que más parece una señal (la probabilidad más alta
        # sin contar SIN_SENAL); los demás suelen ser pilas del robot, etc.
        es_senal = [i for i, c in enumerate(self.clases) if c != "SIN_SENAL"]
        mejor = int(np.argmax(probs[:, es_senal].max(axis=1)))
        self.ultimas_probs = probs[mejor]
        return {c: float(p) for c, p in zip(self.clases, probs[mejor])}


def buscar_modelo(modelo=None, carpeta=None):
    """
    Devuelve (carpeta, seleccion) de la primera carpeta de
    config_robot.CARPETAS_MODELO (o `carpeta`, si se da) que tenga
    seleccion.json y el .keras del modelo elegido.
    """
    candidatas = [carpeta] if carpeta else config_robot.CARPETAS_MODELO
    for c in candidatas:
        ruta = os.path.join(c, "seleccion.json")
        if not os.path.exists(ruta):
            continue
        with open(ruta, encoding="utf-8") as f:
            seleccion = json.load(f)
        nombre = modelo or seleccion["mejores"][0]["modelo"]
        if os.path.exists(os.path.join(c, nombre, f"{nombre}.keras")):
            return c, seleccion
    raise FileNotFoundError(
        f"No hay modelo entrenado en {candidatas}. Entrena con: python entrenar.py "
        "(o copia la carpeta de resultados con los .keras)")


if __name__ == "__main__":
    import sys
    p = PredictorSenales()
    clase, prob = p.predecir(sys.argv[1])
    print(f"[{p.nombre}] {clase} ({prob * 100:.1f}%)")
