import cv2
import numpy as np

from comunicacion_robot import Accion


class SeguidorLinea:
    """
    Sigue la línea negra del piso leyendo su posición en dos
    bandas horizontales del frame: una banda "lejana" (más arriba
    en la imagen, lo que el robot va a encontrar más adelante) y
    una banda "cercana" (justo por encima del frente del robot).

    Etapas del algoritmo (todas vistas en clase):
      1. Recorte de dos regiones de interés (bandas horizontales).
      2. Segmentación por color en HSV: se queda con los píxeles
         oscuros y poco saturados (la línea negra).
      3. Cierre morfológico (rellena los huecos del trazo) y
         apertura con un kernel del grosor mínimo de la línea:
         borra todo lo más delgado que la pista (cables, rayas,
         ruido suelto).
      4. Contornos + centroide (momentos) del contorno más grande
         de cada banda que la cruce de arriba a abajo (la pista
         atraviesa la banda; la cinta negra horizontal de los
         carteles no), que se asume es la línea.
      5. Error respecto al centro del robot: se combinan los dos
         centroides (el lejano anticipa las curvas) y se compara
         contra el centro de la imagen. Como la línea es
         continua, una detección que salta lejos (de una banda
         a la otra, o respecto al frame anterior) se toma como
         otra cosa (cinta de un cartel, sombra) y se descarta.
      6. Decisión: ADELANTE si el error está dentro de la
         tolerancia, IZQUIERDA o DERECHA si no. Si la línea se
         pierde, se sigue girando hacia el último lado donde se
         vio, para recuperarla en vez de descarrilarse.
      7. Filtro temporal: la acción solo cambia cuando la nueva
         decisión se repite varios frames seguidos, para que el
         ruido de un frame suelto no haga zigzaguear al robot.
    """

    COLOR_BANDA_LEJOS = (0, 200, 255)
    COLOR_BANDA_CERCA = (255, 200, 0)
    COLOR_CENTROIDE = (0, 0, 255)
    COLOR_CENTRO_ROBOT = (0, 255, 0)
    COLOR_OBJETIVO = (255, 0, 255)
    COLOR_TEXTO = (255, 255, 255)

    def __init__(self, tamano_kernel_cierre=(5, 5)):
        self.kernel_cierre = np.ones(tamano_kernel_cierre, np.uint8)

        # Última acción decidida mientras la línea se veía. Es la
        # "memoria" que se usa cuando la línea desaparece.
        self.ultima_accion_vista = None

        # Último punto objetivo válido, para rechazar detecciones
        # que aparecen de golpe lejos de donde estaba la línea.
        self.ultimo_objetivo = None

        # Estado del filtro temporal (paso 7).
        self.accion_confirmada = None
        self.accion_candidata = None
        self.frames_candidata = 0

    def procesar(self, frame, parametros, lienzo=None):
        """
        Recibe el frame original (BGR), sobre el que se hace el
        análisis, y un diccionario con los parámetros ya leídos
        de los trackbars (ver InterfazControl.leer_controles_linea).
        Los dibujos se hacen sobre una copia de `lienzo` (o del
        frame si no se pasa). Devuelve una tupla
        (frame_dibujado, accion), donde accion es una de las
        constantes de Accion.
        """
        salida = (frame if lienzo is None else lienzo).copy()
        alto_frame, ancho_frame = frame.shape[:2]

        banda_lejos = self._recortar_banda(
            frame, alto_frame,
            parametros["pos_lejos_pct"], parametros["alto_banda_pct"]
        )
        banda_cerca = self._recortar_banda(
            frame, alto_frame,
            parametros["pos_cerca_pct"], parametros["alto_banda_pct"]
        )

        cx_lejos, contorno_lejos = self._centroide_linea(banda_lejos["imagen"], parametros)
        cx_cerca, contorno_cerca = self._centroide_linea(banda_cerca["imagen"], parametros)

        centro_robot = int(ancho_frame * parametros["centro_pct"] / 100)
        separacion_maxima = ancho_frame * parametros["separacion_max_pct"] / 100
        cerca_ok, lejos_ok = self._validar_bandas(cx_cerca, cx_lejos, separacion_maxima)
        if not cerca_ok:
            cx_cerca, contorno_cerca = None, None
        if not lejos_ok:
            cx_lejos, contorno_lejos = None, None

        objetivo = self._punto_objetivo(cx_cerca, cx_lejos, parametros["peso_lejos_pct"])
        if objetivo is not None:
            self.ultimo_objetivo = objetivo

        tolerancia = ancho_frame * parametros["tolerancia_pct"] / 100
        accion, error = self._decidir_accion(objetivo, centro_robot, tolerancia)
        accion = self._filtrar(accion, parametros["frames_confirmacion"])

        self._dibujar_banda(salida, banda_lejos, contorno_lejos, cx_lejos, self.COLOR_BANDA_LEJOS)
        self._dibujar_banda(salida, banda_cerca, contorno_cerca, cx_cerca, self.COLOR_BANDA_CERCA)
        self._dibujar_guia(salida, centro_robot, tolerancia, objetivo, banda_cerca["y1"])
        self._dibujar_texto(salida, accion, error)

        return salida, accion

    def _recortar_banda(self, frame, alto_frame, pos_pct, alto_pct):
        # Recorte de región de interés: una franja horizontal del
        # frame, ubicada y dimensionada como porcentaje del alto
        # total (para que funcione igual sin importar la
        # resolución del video).
        y0 = int(alto_frame * pos_pct / 100)
        alto = max(int(alto_frame * alto_pct / 100), 1)
        y1 = min(y0 + alto, alto_frame)
        return {"imagen": frame[y0:y1, :], "y0": y0, "y1": y1}

    def _centroide_linea(self, imagen_banda, parametros):
        if imagen_banda.size == 0:
            return None, None

        # Segmentación por color en HSV: la línea es negra/gris
        # (saturación baja, valor bajo), mientras que los carteles
        # PARE/SIGA son rojo/verde saturados. Filtrando por
        # saturación y valor máximos evitamos confundir un cartel
        # en sombra (oscuro en escala de grises) con la línea.
        hsv = cv2.cvtColor(imagen_banda, cv2.COLOR_BGR2HSV)
        binaria = cv2.inRange(
            hsv,
            (0, 0, 0),
            (179, parametros["umbral_saturacion"], parametros["umbral_valor"])
        )

        # Cierre: rellena los huequitos claros que deja el trazo
        # (la línea está pintada a mano) para tener una región
        # sólida.
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, self.kernel_cierre)

        # Apertura con un kernel del grosor mínimo de la línea:
        # todo lo que sea más delgado que ese kernel (un cable,
        # una raya, ruido) desaparece; la pista, que es gruesa,
        # sobrevive.
        grosor = max(parametros["grosor_min"], 1)
        kernel_grosor = np.ones((grosor, grosor), np.uint8)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_grosor)

        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None, None

        # La pista cruza la banda de arriba a abajo; una franja
        # horizontal (la cinta negra de los carteles) o un pedazo
        # suelto no. Se descartan los contornos cuyo alto no
        # cubra un porcentaje mínimo del alto de la banda.
        alto_minimo = imagen_banda.shape[0] * parametros["cruce_min_pct"] / 100
        contornos = [c for c in contornos if cv2.boundingRect(c)[3] >= alto_minimo]
        if not contornos:
            return None, None

        contorno_mayor = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(contorno_mayor) < parametros["area_minima"]:
            return None, None

        momentos = cv2.moments(contorno_mayor)
        if momentos["m00"] == 0:
            return None, None

        cx = int(momentos["m10"] / momentos["m00"])
        return cx, contorno_mayor

    def _validar_bandas(self, cx_cerca, cx_lejos, separacion_maxima):
        # Devuelve (usar_cerca, usar_lejos). La línea es continua:
        # no salta de un lado al otro entre bandas ni entre un
        # frame y el siguiente. Una detección que rompe esa
        # continuidad suele ser otra cosa (la cinta de un cartel,
        # una sombra) y se descarta.
        if cx_cerca is not None and cx_lejos is not None:
            if abs(cx_lejos - cx_cerca) <= separacion_maxima:
                return True, True
            # Las bandas no coinciden: se le cree a la que quede
            # más cerca de donde estaba la línea antes (sin
            # historia, a la cercana, que está justo frente al
            # robot).
            if self.ultimo_objetivo is None:
                return True, False
            if abs(cx_cerca - self.ultimo_objetivo) <= abs(cx_lejos - self.ultimo_objetivo):
                return True, False
            return False, True

        # Solo una banda (o ninguna) vio algo: se acepta si queda
        # cerca del último objetivo válido.
        return self._es_continuo(cx_cerca, separacion_maxima), \
            self._es_continuo(cx_lejos, separacion_maxima)

    def _es_continuo(self, cx, separacion_maxima):
        if cx is None:
            return False
        if self.ultimo_objetivo is None:
            return True
        return abs(cx - self.ultimo_objetivo) <= separacion_maxima

    def _punto_objetivo(self, cx_cerca, cx_lejos, peso_lejos_pct):
        # Punto al que el robot debería apuntar: promedio
        # ponderado de los dos centroides. Darle peso a la banda
        # lejana hace que el robot empiece a girar antes de que
        # la curva le llegue encima. Si solo una banda ve la
        # línea, se usa esa sola.
        if cx_cerca is None and cx_lejos is None:
            return None
        if cx_lejos is None:
            return cx_cerca
        if cx_cerca is None:
            return cx_lejos

        peso = peso_lejos_pct / 100
        return int((1 - peso) * cx_cerca + peso * cx_lejos)

    def _decidir_accion(self, objetivo, centro_robot, tolerancia):
        # Línea perdida: repetir lo último que se hizo con la
        # línea a la vista. Si se salió por un lado, el robot
        # sigue girando hacia ese lado hasta reencontrarla; si
        # iba centrada (un cartel la tapa, por ejemplo), sigue
        # derecho. Si nunca se vio, lo seguro es quedarse quieto.
        if objetivo is None:
            return self.ultima_accion_vista or Accion.PARAR, None

        # Error: cuántos píxeles está la línea corrida respecto
        # al centro del robot (negativo = a la izquierda).
        error = objetivo - centro_robot

        if error < -tolerancia:
            accion = Accion.IZQUIERDA
        elif error > tolerancia:
            accion = Accion.DERECHA
        else:
            accion = Accion.ADELANTE

        self.ultima_accion_vista = accion
        return accion, error

    def _filtrar(self, accion, frames_confirmacion):
        # La primera decisión se acepta directo; después, una
        # acción distinta tiene que repetirse
        # `frames_confirmacion` frames seguidos para reemplazar
        # a la actual.
        if self.accion_confirmada is None or accion == self.accion_confirmada:
            self.accion_confirmada = accion
            self.accion_candidata = None
            self.frames_candidata = 0
            return self.accion_confirmada

        if accion == self.accion_candidata:
            self.frames_candidata += 1
        else:
            self.accion_candidata = accion
            self.frames_candidata = 1

        if self.frames_candidata >= frames_confirmacion:
            self.accion_confirmada = accion
            self.accion_candidata = None
            self.frames_candidata = 0

        return self.accion_confirmada

    def _dibujar_banda(self, salida, banda, contorno, cx, color):
        y0, y1 = banda["y0"], banda["y1"]
        cv2.rectangle(salida, (0, y0), (salida.shape[1] - 1, y1), color, 1)

        if contorno is not None:
            contorno_trasladado = contorno + [0, y0]
            cv2.drawContours(salida, [contorno_trasladado], -1, color, 2)

        if cx is not None:
            cy = (y0 + y1) // 2
            cv2.circle(salida, (cx, cy), 6, self.COLOR_CENTROIDE, -1)

    def _dibujar_guia(self, salida, centro_robot, tolerancia, objetivo, y):
        # Línea vertical verde = centro del robot; las dos
        # marcas a los lados = zona de tolerancia (ADELANTE).
        alto = salida.shape[0]
        cv2.line(salida, (centro_robot, 0), (centro_robot, alto), self.COLOR_CENTRO_ROBOT, 1)
        for x in (int(centro_robot - tolerancia), int(centro_robot + tolerancia)):
            cv2.line(salida, (x, y - 15), (x, y + 15), self.COLOR_CENTRO_ROBOT, 2)

        if objetivo is not None:
            cv2.circle(salida, (objetivo, y), 8, self.COLOR_OBJETIVO, 2)

    def _dibujar_texto(self, salida, accion, error):
        texto = f"linea: {accion}" if error is not None else f"linea: {accion} (perdida)"
        self._poner_texto(salida, texto, (10, 30), 0.8)
        if error is not None:
            self._poner_texto(salida, f"error: {error:+d} px", (10, 60), 0.6)

    def _poner_texto(self, salida, texto, posicion, escala):
        # Texto blanco con borde negro para que se lea tanto
        # sobre el piso claro como sobre la línea oscura.
        for color, grosor in (((0, 0, 0), 4), (self.COLOR_TEXTO, 2)):
            cv2.putText(
                salida, texto, posicion,
                cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA
            )
