import cv2
import numpy as np


class ClasificadorColor:
    """
    Decide de qué color es el interior de un contorno, usando
    K-Means sobre los píxeles que quedan dentro de él.

    K-Means (en su modalidad básica) se usa solo para esta
    decisión puntual: separar el color de fondo del octágono
    del blanco de las letras "PARE"/"SIGA". La FORMA la
    analiza DetectorSenales.
    """

    def __init__(self, minimo_pixeles=30):
        self.minimo_pixeles = minimo_pixeles
        self.criterios_kmeans = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0
        )

    def _mascara_del_contorno(self, forma_frame, contorno):
        mascara = np.zeros(forma_frame, dtype=np.uint8)
        cv2.drawContours(mascara, [contorno], -1, 255, -1)
        return mascara

    def _pixeles_hsv_dentro(self, frame_bgr, mascara):
        # AND entre frame y máscara (Diapositiva 7)
        recorte_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mascara)
        # Transformación BGR a HSV (Diapositivas 4 y 5)
        recorte_hsv = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2HSV)
        return recorte_hsv[mascara == 255].astype(np.float32)

    def _color_dominante_hsv(self, pixeles_hsv):
        # K-Means con k=2 para separar el fondo de letras o reflejos
        k = 2 if len(pixeles_hsv) >= 30 else 1
        _, etiquetas, centros = cv2.kmeans(
            pixeles_hsv,
            k,
            None,
            self.criterios_kmeans,
            5,
            cv2.KMEANS_RANDOM_CENTERS
        )

        conteos = np.bincount(etiquetas.flatten())
        # Tomamos el cluster con más píxeles (color dominante de la señal)
        indice_dominante = np.argmax(conteos)
        return centros[indice_dominante]

    def _nombre_por_matiz(self, h, s):
        # En OpenCV, H va de 0 a 179. El rojo queda cerca de los
        # extremos (0 o 179); el verde en el medio (40-85 aprox).
        # Exigimos saturación alta para no confundir blanco/gris
        # (que tienen S baja) con un color real.
        if (h <= 10 or h >= 170) and s > 80:
            return "PARE"
        elif 40 <= h <= 85 and s > 80:
            return "SIGA"
        return "DESCONOCIDO"
