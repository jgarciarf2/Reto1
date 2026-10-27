import cv2
import numpy as np


class Preprocesador:
    """
    Prepara el frame para buscar las señales PARE/SIGA: devuelve
    máscaras binarias con los píxeles que son rojo o verde
    saturado, dentro de la zona del frame donde pueden aparecer
    los carteles.

        recorte ROI -> blur gaussiano -> HSV -> umbral por color
        (rojo OR verde) -> apertura + cierre

    Esta clase no decide qué es cada mancha; solo entrega las
    máscaras para que DetectorSenales analice la forma.
    """

    # En OpenCV el matiz (H) va de 0 a 179. El rojo queda en los
    # dos extremos del rango, por eso necesita dos intervalos.
    RANGOS_ROJO = ((0, 10), (160, 179))
    RANGO_VERDE = (40, 90)

    def __init__(self, tamano_kernel_morfologico=(5, 5)):
        self.kernel_morfologico = np.ones(tamano_kernel_morfologico, np.uint8)

    def mascaras_senales(self, frame, parametros):
        """
        Devuelve un diccionario con tres máscaras del tamaño del
        frame: "roja", "verde" y "total" (rojo OR verde).
        """
        alto = frame.shape[0]

        # Recorte de la región de interés: la parte de abajo del
        # frame es el propio robot (tiene piezas verdes, rojas y
        # amarillas que confundirían al detector).
        limite = int(alto * parametros["roi_pct"] / 100)
        roi = frame[:limite]

        suavizado = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(suavizado, cv2.COLOR_BGR2HSV)

        sat_min = parametros["sat_min"]
        val_min = parametros["val_min"]

        # Umbralización por color: solo pasan los píxeles con
        # matiz rojo o verde Y bien saturados (los grises, el
        # blanco del piso y la línea negra tienen saturación baja).
        roja = np.zeros(hsv.shape[:2], np.uint8)
        for h_min, h_max in self.RANGOS_ROJO:
            rango = cv2.inRange(hsv, (h_min, sat_min, val_min), (h_max, 255, 255))
            roja = cv2.bitwise_or(roja, rango)

        h_min, h_max = self.RANGO_VERDE
        verde = cv2.inRange(hsv, (h_min, sat_min, val_min), (h_max, 255, 255))

        mascaras = {}
        for nombre, mascara in (("roja", roja), ("verde", verde)):
            # Apertura (quita puntos sueltos) y cierre (rellena
            # las letras blancas de adentro del cartel).
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, self.kernel_morfologico)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, self.kernel_morfologico)

            # Se devuelve del tamaño del frame completo (la parte
            # fuera de la ROI queda en negro) para que las
            # coordenadas de los contornos coincidan con el frame.
            completa = np.zeros(frame.shape[:2], np.uint8)
            completa[:limite] = mascara
            mascaras[nombre] = completa

        # Operación OR: una sola máscara con las dos, que es sobre
        # la que se buscan las formas.
        mascaras["total"] = cv2.bitwise_or(mascaras["roja"], mascaras["verde"])
        return mascaras

    def area_mayor(self, mascara):
        """Área del contorno más grande de la máscara (0 si no hay)."""
        contornos = self.encontrar_contornos(mascara)
        if not contornos:
            return 0
        return max(cv2.contourArea(c) for c in contornos)

    def encontrar_contornos(self, mascara):
        contornos, _ = cv2.findContours(
            mascara,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contornos
