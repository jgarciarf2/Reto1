import cv2


class InterfazControl:
    """
    Encapsula las ventanas de OpenCV: una solo para ver el video
    y otra con los trackbars (controles deslizantes) del seguidor
    de línea, que permiten ajustar sus parámetros en vivo sin
    tener que tocar el código cada vez.

    Los trackbars se separan del video (si comparten ventana, el
    video queda apretado en una franja chiquita). OpenCV además
    recorta en pantalla los nombres de trackbar más largos que el
    primero que se creó en esa ventana, así que los nombres se
    mantienen cortos a propósito; el significado de cada uno se
    explica en _imprimir_leyenda().
    Las señales ya no tienen trackbars: las reconoce la CNN.
    """

    def __init__(self, nombre_ventana="Robot - Linea + CNN"):
        self.nombre_ventana = nombre_ventana
        self.nombre_ventana_controles_linea = "Controles - Linea"

        cv2.namedWindow(self.nombre_ventana, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.nombre_ventana_controles_linea, cv2.WINDOW_NORMAL)

        cv2.resizeWindow(self.nombre_ventana, 720, 640)
        cv2.resizeWindow(self.nombre_ventana_controles_linea, 420, 480)

        self._crear_trackbars_linea()
        self._imprimir_leyenda()

    def _nada(self, valor):
        # Callback vacío que exige la API de trackbars de OpenCV
        # (siempre espera una función, aunque no haga nada).
        pass

    def _crear_trackbars_linea(self):
        # Controles del seguidor de línea: umbral de color (HSV) y
        # ubicación de las dos bandas de lectura, todo en
        # porcentaje para que funcione igual sin importar la
        # resolución del video.
        ventana = self.nombre_ventana_controles_linea
        cv2.createTrackbar("Umbral Val", ventana, 120, 255, self._nada)
        cv2.createTrackbar("Umbral Sat", ventana, 60, 255, self._nada)
        cv2.createTrackbar("Pos Lejos", ventana, 12, 100, self._nada)
        cv2.createTrackbar("Pos Cerca", ventana, 40, 100, self._nada)
        cv2.createTrackbar("Alto Banda", ventana, 10, 100, self._nada)
        cv2.createTrackbar("Area Min", ventana, 300, 5000, self._nada)
        cv2.createTrackbar("Centro", ventana, 50, 100, self._nada)
        cv2.createTrackbar("Tolerancia", ventana, 10, 50, self._nada)
        cv2.createTrackbar("Peso Lejos", ventana, 50, 100, self._nada)
        cv2.createTrackbar("Grosor Min", ventana, 15, 60, self._nada)
        cv2.createTrackbar("Separ Max", ventana, 45, 100, self._nada)
        cv2.createTrackbar("Cruce Min", ventana, 90, 100, self._nada)
        cv2.createTrackbar("Confirmar", ventana, 3, 15, self._nada)

    def _imprimir_leyenda(self):
        print("=== Controles - Linea (seguidor de linea negra) ===")
        print("  Umbral Val / Umbral Sat : limites de Valor y Saturacion (HSV) para pintar un pixel como 'linea'")
        print("  Pos Lejos / Pos Cerca   : posicion (%) de cada banda de lectura, desde arriba del frame")
        print("  Alto Banda              : alto (%) de cada banda de lectura")
        print("  Area Min                : area minima del contorno de linea dentro de una banda")
        print("  Centro                  : posicion (%) del centro del robot en el ancho del frame")
        print("  Tolerancia              : margen (%) del ancho alrededor del centro en el que se sigue ADELANTE")
        print("  Peso Lejos              : peso (%) de la banda lejana al calcular el punto objetivo")
        print("  Grosor Min              : grosor minimo (px) de la linea; lo mas delgado se borra con apertura")
        print("  Separ Max               : separacion maxima (%) entre centroides antes de ignorar la banda lejana")
        print("  Cruce Min               : alto minimo (%) de la banda que debe cubrir un contorno para ser la linea")
        print("  Confirmar               : frames seguidos que debe repetirse una accion nueva antes de enviarla")

    def leer_controles_linea(self):
        """
        Devuelve, como diccionario, los parámetros actuales de
        los trackbars del seguidor de línea.
        """
        ventana = self.nombre_ventana_controles_linea
        return {
            "umbral_valor": cv2.getTrackbarPos("Umbral Val", ventana),
            "umbral_saturacion": cv2.getTrackbarPos("Umbral Sat", ventana),
            "pos_lejos_pct": cv2.getTrackbarPos("Pos Lejos", ventana),
            "pos_cerca_pct": cv2.getTrackbarPos("Pos Cerca", ventana),
            "alto_banda_pct": max(cv2.getTrackbarPos("Alto Banda", ventana), 1),
            "area_minima": cv2.getTrackbarPos("Area Min", ventana),
            "centro_pct": cv2.getTrackbarPos("Centro", ventana),
            "tolerancia_pct": cv2.getTrackbarPos("Tolerancia", ventana),
            "peso_lejos_pct": cv2.getTrackbarPos("Peso Lejos", ventana),
            "grosor_min": cv2.getTrackbarPos("Grosor Min", ventana),
            "separacion_max_pct": cv2.getTrackbarPos("Separ Max", ventana),
            "cruce_min_pct": cv2.getTrackbarPos("Cruce Min", ventana),
            "frames_confirmacion": max(cv2.getTrackbarPos("Confirmar", ventana), 1),
        }
