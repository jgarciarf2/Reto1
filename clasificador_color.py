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

        # Criterio de parada de K-Means: parar a las 10
        # iteraciones o antes si el cambio entre iteraciones es
        # menor a 1.0.
        self.criterios_kmeans = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0
        )

    def clasificar(self, frame_bgr, contorno_aproximado):
        """
        Devuelve ("PARE" | "SIGA" | "DESCONOCIDO", (h, s, v))
        o (None, None) si no hay píxeles suficientes.
        """
        mascara = self._mascara_del_contorno(frame_bgr.shape[:2], contorno_aproximado)
        pixeles = self._pixeles_hsv_dentro(frame_bgr, mascara)

        if len(pixeles) < self.minimo_pixeles:
            return None, None

        h, s, v = self._color_dominante(pixeles)
        etiqueta = self._nombre_por_matiz(h, s)

        return etiqueta, (h, s, v)

    def _mascara_del_contorno(self, forma_frame, contorno):
        # Imagen negra del tamaño del frame, rellena de blanco
        # (255) solo en la región del contorno.
        mascara = np.zeros(forma_frame, dtype=np.uint8)
        cv2.drawContours(mascara, [contorno], -1, 255, -1)
        return mascara

    def _pixeles_hsv_dentro(self, frame_bgr, mascara):
        # Operación AND: el frame AND la máscara deja pasar
        # únicamente los píxeles de adentro del contorno.
        recorte = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mascara)

        # HSV separa el "tipo de color" (H) del brillo/saturación,
        # lo que hace más fácil y estable decidir rojo vs verde.
        recorte_hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)

        return recorte_hsv[mascara == 255].astype(np.float32)

    def _color_dominante(self, pixeles):
        # k=2: un cluster para el fondo del octágono, otro para
        # el blanco de las letras. Nos quedamos con el más
        # grande (más píxeles), que es el fondo.
        k = 2
        _, etiquetas, centros = cv2.kmeans(
            pixeles,
            k,
            None,
            self.criterios_kmeans,
            5,                          # intentos con distintos puntos de partida
            cv2.KMEANS_RANDOM_CENTERS
        )

        conteos = np.bincount(etiquetas.flatten())
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
