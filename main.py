import time

import cv2

import config_robot
from comunicacion_robot import Accion, ComunicacionRobot, SalidaConsola
from control_cnn import ControlCNN
from hilos import CamaraEnVivo, HiloCNN
from interfaz_control import InterfazControl
from predictor_senales import PredictorSenales
from seguidor_linea import SeguidorLinea
from visualizador import Visualizador


# ============================================================
# IDEA GENERAL
# ============================================================
# En cada frame de la cámara el programa hace dos cosas en
# paralelo y al final decide una sola acción para el robot:
#
#   SeguidorLinea     -> mide qué tan corrida está la línea negra
#                         respecto al centro del robot y propone
#                         ADELANTE, IZQUIERDA o DERECHA.
#   PredictorSenales  -> la CNN entrenada (reto 2) mira el frame y
#                         dice qué señal ve: PARE, SIGA,
#                         VELOCIDAD_60 o ZONA_ESCOLAR.
#   ControlCNN        -> confirma la señal (promedio de varias
#                         predicciones) y decide: PARE detiene al
#                         robot hasta ver SIGA; ZONA_ESCOLAR baja a
#                         velocidad REDUCIDA y VELOCIDAD_60 vuelve a
#                         la NORMAL (ninguna de las dos saca al robot
#                         de un PARE).
#   ComunicacionRobot -> traduce la acción a la letra que
#                         entiende el robot (config_robot.py) y
#                         la envía.
#   Visualizador      -> arma el panel que se ve en pantalla.
#   InterfazControl   -> ventanas + trackbars de la línea.
#
# Las señales ya no se detectan por forma/color (reto 1): esa
# versión sigue en la rama feature/reto-completo para comparar.
# En la terminal solo se imprime la señal que detecta la CNN.


class AplicacionDetector:
    """
    Punto de entrada del programa: abre el archivo de video o cámara,
    arma cada pieza del pipeline y corre el loop principal (leer frame,
    procesarlo, mostrarlo) hasta que el usuario presione 'q'.
    """

    COLOR_SIGUIENDO = (0, 200, 0)
    COLOR_DETENIDO = (0, 0, 255)
    # Grados que se gira el frame para verlo derecho (0, 90, -90 o 180).
    # La webcam del computador va en 0; el stream del móvil suele necesitar 90.
    # Se cambia en config_robot.ROTACION_CAMARA o con --rotar=90.
    ROTACIONES = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
    }
    # La CNN es la parte pesada: se consulta cada INTERVALO_CNN_S, no en cada frame.
    INTERVALO_PREDICCION_S = config_robot.INTERVALO_CNN_S

    def __init__(self, fuente_video="0", sin_robot=False, rotacion=None, confianza=None):
        self.fuente_video = fuente_video
        grados = config_robot.ROTACION_CAMARA if rotacion is None else rotacion
        self.rotacion = self.ROTACIONES[grados]
        # Un número ("0", "1", ...) es el índice de la cámara física;
        # cualquier otra cosa es una ruta a un archivo de video.
        indice = int(fuente_video) if str(fuente_video).isdigit() else fuente_video
        self.es_camara = isinstance(indice, int) or str(indice).startswith("http")

        try:
            self.predictor = PredictorSenales()
        except FileNotFoundError as e:
            raise SystemExit(str(e))
        print(f"CNN cargada: {self.predictor.nombre} ({self.predictor.carpeta}), "
              f"clases {self.predictor.clases}")
        self.control_cnn = ControlCNN(confianza_minima=confianza)

        # Primero el robot: conectarse por Bluetooth puede tardar
        # varios segundos, y si las ventanas ya estuvieran abiertas
        # se quedarían grises y congeladas mientras tanto.
        # En modo prueba no se conecta.
        if sin_robot:
            self.robot = ComunicacionRobot(salida=SalidaConsola())
        else:
            self.robot = ComunicacionRobot()

        self.cap = self._abrir_camara(indice)

        self.interfaz = InterfazControl()
        self.seguidor_linea = SeguidorLinea()
        self.visualizador = Visualizador()
        self.ultima_prediccion = 0.0
        self.texto_cnn = "CNN..."
        self.lineas_probs = []
        self.hilo_cnn = None  # se crea en ejecutar(); sin él la CNN corre en el loop
        self.fps = 0.0

    def _abrir_camara(self, indice):
        cap = cv2.VideoCapture(indice)
        if self.es_camara and not cap.isOpened():
            # El iPhone (Cámara de continuidad) puede tardar en aparecer: se reintenta ~15 s.
            print("Esperando la cámara... (desbloquea el iPhone y acércalo al Mac)")
            for _ in range(15):
                time.sleep(1)
                cap = cv2.VideoCapture(indice)
                if cap.isOpened():
                    break
        return cap

    def ejecutar(self):
        if not self.cap.isOpened():
            print(f"No se pudo abrir el video/camara: {self.fuente_video}")
            return

        # En paralelo (ver hilos.py): la cámara se lee en su propio hilo
        # (siempre el frame más nuevo, sin atraso) y la CNN corre en otro.
        if self.es_camara:
            self.cap = CamaraEnVivo(self.cap)
        self.hilo_cnn = HiloCNN(self.predictor, self.control_cnn,
                                self.INTERVALO_PREDICCION_S, self._avisar_senal)
        # Un archivo de video se reproduce a su velocidad; la cámara, lo más rápido posible
        espera_ms = 1 if self.es_camara else 30
        inicio_fps, frames_fps = time.time(), 0

        try:
            while True:
                ret, frame = self._leer_frame()
                if not ret:
                    print("No llegan imágenes de la cámara/video.")
                    break

                # El seguidor de línea necesita el robot abajo (frame girado);
                # la CNN funciona mejor con el frame tal como sale de la cámara.
                frame_cnn = frame
                if self.rotacion is not None:
                    frame = cv2.rotate(frame, self.rotacion)
                panel = self._procesar_frame(frame, frame_cnn)
                self.interfaz.mostrar(panel)

                frames_fps += 1
                if time.time() - inicio_fps >= 1.0:
                    self.fps = frames_fps / (time.time() - inicio_fps)
                    inicio_fps, frames_fps = time.time(), 0

                if cv2.waitKey(espera_ms) & 0xFF == ord("q"):
                    break
        finally:
            self.hilo_cnn.detener()
            self.robot.cerrar()
            self.cap.release()
            cv2.destroyAllWindows()

    def _leer_frame(self):
        if self.es_camara:
            # CamaraEnVivo ya espera hasta ~5 s si la cámara tarda en arrancar
            return self.cap.read()

        ret, frame = self.cap.read()
        if ret:
            return ret, frame

        # Si finaliza el video, reiniciar al inicio para bucle
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self.cap.read()

    def _procesar_frame(self, frame, frame_cnn=None):
        # 1. Línea: acción propuesta por el seguidor.
        salida, accion_linea = self.seguidor_linea.procesar(
            frame, self.interfaz.leer_controles_linea())

        # 2. Señal: la CNN decide si hay que parar o cambiar la velocidad.
        frame_cnn = frame if frame_cnn is None or not config_robot.CNN_SIN_ROTAR else frame_cnn
        if self.hilo_cnn:
            self.hilo_cnn.ver(frame_cnn)  # la CNN lo procesa en su hilo
        else:
            self._consultar_cnn(frame_cnn)

        # 3. Decisión final y envío al robot.
        accion = Accion.PARAR if self.control_cnn.detenido else accion_linea
        self.robot.velocidad = self.control_cnn.velocidad
        self.robot.enviar(accion)

        self._dibujar_estado(salida, accion)
        return self.visualizador.crear_panel(salida)

    def _consultar_cnn(self, frame):
        ahora = time.time()
        if ahora - self.ultima_prediccion < self.INTERVALO_PREDICCION_S:
            return
        self.ultima_prediccion = ahora
        probs = self.predictor.probabilidades(frame)
        clase = max(probs, key=probs.get)
        self.texto_cnn = f"CNN: {clase} {probs[clase] * 100:.0f}%"
        self.lineas_probs = [f"{c} {p * 100:.0f}%" for c, p in probs.items()]
        nueva = self.control_cnn.actualizar(probs)
        if nueva:
            self._avisar_senal(nueva)

    def _avisar_senal(self, senal):
        # única salida por terminal: la señal detectada y cómo queda el robot
        estado = "DETENIDO" if self.control_cnn.detenido else "EN MARCHA"
        print(f"{senal} -> {estado}, velocidad {self.control_cnn.velocidad}")

    def _dibujar_estado(self, salida, accion):
        if self.hilo_cnn:
            self.texto_cnn = self.hilo_cnn.texto
            self.lineas_probs = self.hilo_cnn.lineas_probs
        cv2.putText(salida, self.texto_cnn, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(salida, self.texto_cnn, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2, cv2.LINE_AA)

        # Probabilidad de cada clase (sirve para ver si SIGA/ZONA se acercan al umbral)
        for i, linea in enumerate(self.lineas_probs):
            y_linea = 125 + i * 28
            cv2.putText(salida, linea, (10, y_linea), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(salida, linea, (10, y_linea), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        detenido = self.control_cnn.detenido
        texto = (f"{'DETENIDO' if detenido else 'SIGUIENDO'} -> {accion}"
                 f"  vel {self.control_cnn.velocidad}")
        y = salida.shape[0] - 20
        cv2.putText(salida, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(salida, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    self.COLOR_DETENIDO if detenido else self.COLOR_SIGUIENDO, 2, cv2.LINE_AA)

        # Rendimiento: frames por segundo del seguidor y lo que tarda la CNN
        ms_cnn = self.hilo_cnn.ms_por_prediccion if self.hilo_cnn else 0
        texto = f"{self.fps:.0f} fps   CNN {ms_cnn:.0f} ms"
        cv2.putText(salida, texto, (10, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(salida, texto, (10, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    import sys
    # Uso: python main.py [fuente] [--prueba] [--rotar=90] [--confianza=0.6]
    #   python main.py 0 --prueba            -> webcam 0, sin robot
    #   python main.py 2 --prueba --rotar=90 -> cámara 2 girada 90°, sin robot
    #   python main.py 2                     -> cámara 2, conectado al robot
    argumentos = [a for a in sys.argv[1:] if not a.startswith("--")]
    sin_robot = "--prueba" in sys.argv
    ruta_video = argumentos[0] if argumentos else "0"
    rotacion = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--rotar=")), None)
    confianza = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--confianza=")), None)
    AplicacionDetector(fuente_video=ruta_video, sin_robot=sin_robot,
                       rotacion=rotacion, confianza=confianza).ejecutar()
