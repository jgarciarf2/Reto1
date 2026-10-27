import cv2


class Visualizador:
    """
    Arma el panel que se muestra en pantalla: la salida con todo
    dibujado, redimensionada manteniendo la proporción del video
    para que un video vertical no se vea aplastado.
    """

    def __init__(self, alto_panel=640):
        self.alto_panel = alto_panel

    def crear_panel(self, salida):
        alto, ancho = salida.shape[:2]
        escala = self.alto_panel / alto
        return cv2.resize(salida, (int(ancho * escala), self.alto_panel))
