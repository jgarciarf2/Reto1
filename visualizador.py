import cv2
import numpy as np


class Visualizador:
    """
    Arma el panel que se muestra en pantalla: la salida con todo
    dibujado y, al lado, la máscara de color de las señales
    (sirve para calibrar los trackbars de Sat Min / Val Min).
    Se redimensiona manteniendo la proporción del video, para
    que un video vertical no se vea aplastado.
    """

    def __init__(self, alto_panel=640):
        self.alto_panel = alto_panel

    def crear_panel(self, salida, mascara):
        mascara_bgr = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
        # return np.hstack([self._redimensionar(salida), self._redimensionar(mascara_bgr)])
        return np.hstack([self._redimensionar(salida)])
        

    def _redimensionar(self, imagen):
        alto, ancho = imagen.shape[:2]
        escala = self.alto_panel / alto
        return cv2.resize(imagen, (int(ancho * escala), self.alto_panel))
