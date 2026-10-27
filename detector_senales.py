import math

import cv2


class DetectorSenales:
    """
    Combina el Preprocesador (máscara de color) con el
    ClasificadorColor (K-Means) para encontrar las señales
    PARE/SIGA entre los contornos de la máscara, y las dibuja
    sobre el frame.

    Un contorno se acepta como señal si:
      - su área supera el mínimo (el cartel está cerca),
      - no toca el borde del frame ni el límite de la ROI (un
        cartel visto a medias no deja confirmar su forma),
      - su aproximación poligonal corresponde a una de las
        FORMAS_VALIDAS: octágono (8 vértices, con margen de 7 a
        9 porque approxPolyDP a veces junta o parte una esquina)
        o cuadrado girado (4 vértices),
      - su relación de aspecto es cercana a 1 (el octágono es
        tan ancho como alto; un cartel cortado por el borde
        del frame no lo es),
      - es compacto: circularidad = 4·pi·área / perímetro² del
        polígono aproximado. Un círculo da 1, un octágono regular
        0.95, un cuadrado 0.785; un rombo aplastado o un
        rectángulo alargado dan mucho menos.
    Después K-Means decide si el interior es rojo o verde.
    """

    COLORES_DIBUJO = {
        "PARE": (0, 0, 255),        # rojo en BGR
        "SIGA": (0, 200, 0),        # verde en BGR
        "DESCONOCIDO": (0, 255, 255)  # amarillo: forma válida pero color dudoso
    }

    # Nombre de la forma -> (vértices mínimos, vértices máximos)
    FORMAS_VALIDAS = {
        "OCTAGONO": (7, 9),
        "CUADRADO": (4, 4),
    }

    ASPECTO_MIN = 0.6
    ASPECTO_MAX = 1.6
    CIRCULARIDAD_MIN = 0.70
    MARGEN_BORDE = 3

    def __init__(self, clasificador_color):
        self.clasificador_color = clasificador_color

    def procesar(self, frame, contornos, parametros):
        """
        Recibe el frame original, los contornos de la máscara de
        color y los parámetros de los trackbars. Devuelve
        (frame_dibujado, detecciones), donde detecciones es una
        lista de diccionarios {"etiqueta", "forma", "area",
        "contorno"}.
        """
        salida = frame.copy()
        detecciones = []

        for contorno in contornos:
            aproximacion, forma = self._aproximar_si_vale(contorno, frame.shape, parametros)

            if aproximacion is None:
                continue

            etiqueta, _hsv = self.clasificador_color.clasificar(frame, aproximacion)

            if etiqueta is None:
                continue

            detecciones.append({
                "etiqueta": etiqueta,
                "forma": forma,
                "area": cv2.contourArea(contorno),
                "contorno": aproximacion,
            })
            self._dibujar(salida, aproximacion, etiqueta, forma)

        return salida, detecciones

    def _aproximar_si_vale(self, contorno, forma_frame, parametros):
        # Devuelve (aproximacion, nombre_forma) o (None, None).
        area = cv2.contourArea(contorno)
        if area < parametros["area_minima"]:
            return None, None

        limite_roi = int(forma_frame[0] * parametros["roi_pct"] / 100)
        if self._toca_borde(contorno, forma_frame[1], limite_roi):
            return None, None

        perimetro = cv2.arcLength(contorno, True)
        epsilon = (parametros["precision"] / 100) * perimetro
        aproximacion = cv2.approxPolyDP(contorno, epsilon, True)

        # Forma: solo octágonos y cuadrados girados. Cualquier
        # otra figura se descarta acá.
        forma = self._nombre_forma(len(aproximacion))
        if forma is None:
            return None, None

        _, _, ancho, alto = cv2.boundingRect(aproximacion)
        aspecto = ancho / alto
        if not self.ASPECTO_MIN <= aspecto <= self.ASPECTO_MAX:
            return None, None

        # Circularidad del polígono aproximado (no del contorno
        # crudo: el borde dentado de la máscara inflaría el
        # perímetro). Solo usa área y perímetro.
        area_poligono = cv2.contourArea(aproximacion)
        perimetro_poligono = cv2.arcLength(aproximacion, True)
        circularidad = 4 * math.pi * area_poligono / (perimetro_poligono ** 2)
        if circularidad < self.CIRCULARIDAD_MIN:
            return None, None

        return aproximacion, forma

    def _nombre_forma(self, vertices):
        for nombre, (minimo, maximo) in self.FORMAS_VALIDAS.items():
            if minimo <= vertices <= maximo:
                return nombre
        return None

    def _toca_borde(self, contorno, ancho, limite_roi):
        x, y, w, h = cv2.boundingRect(contorno)
        m = self.MARGEN_BORDE
        return x <= m or y <= m or x + w >= ancho - m or y + h >= limite_roi - m

    def _dibujar(self, salida, aproximacion, etiqueta, forma):
        x, y, w, h = cv2.boundingRect(aproximacion)
        color = self.COLORES_DIBUJO[etiqueta]

        cv2.drawContours(salida, [aproximacion], -1, color, 3)
        cv2.rectangle(salida, (x, y), (x + w, y + h), color, 2)

        y_texto = y - 10 if y - 10 > 10 else y + h + 25
        cv2.putText(
            salida,
            f"{etiqueta} ({forma.lower()})",
            (x, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )
