import os
import time

import cv2

import config_robot
from comunicacion_robot import Accion, ComunicacionRobot, SalidaConsola
from control_cnn import ControlCNN
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
#   ControlCNN        -> confirma la señal (varias predicciones
#                         seguidas) y decide: PARE detiene al robot;
#                         las demás cambian cada cuánto se envía
#                         "adelante" (más espaciado = más lento).
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
    # El stream del movil llega apaisado aunque el telefono este vertical.
    # Cambia a ROTATE_90_COUNTERCLOCKWISE si la imagen queda invertida.
    ROTACION_CAMARA = cv2.ROTATE_90_CLOCKWISE
    # La CNN es la parte pesada: se consulta ~4 veces por segundo, no en cada frame.
    INTERVALO_PREDICCION_S = 0.25

    def __init__(self, fuente_video="0", sin_robot=False):
        self.fuente_video = fuente_video
        # Un número ("0", "1", ...) es el índice de la cámara física;
        # cualquier otra cosa es una ruta a un archivo de video.
        indice = int(fuente_video) if str(fuente_video).isdigit() else fuente_video

        experimento = config_robot.EXPERIMENTO_CNN
        if not os.path.exists(os.path.join("resultados", experimento, "seleccion.json")):
            raise SystemExit(
                f"No hay modelo entrenado en resultados/{experimento}/. "
                "Entrena con: .venv/bin/python entrenar.py " + experimento)
        self.predictor = PredictorSenales(experimento)
        self.control_cnn = ControlCNN()

        # Primero el robot: conectarse por Bluetooth puede tardar
        # varios segundos, y si las ventanas ya estuvieran abiertas
        # se quedarían grises y congeladas mientras tanto.
        # En modo prueba no se conecta.
        if sin_robot:
            self.robot = ComunicacionRobot(salida=SalidaConsola())
        else:
            self.robot = ComunicacionRobot()

        self.cap = cv2.VideoCapture(indice)

        self.interfaz = InterfazControl()
        self.seguidor_linea = SeguidorLinea()
        self.visualizador = Visualizador()
        self.ultima_prediccion = 0.0
        self.texto_cnn = "CNN..."

    def ejecutar(self):
        if not self.cap.isOpened():
            print(f"No se pudo abrir el video/camara: {self.fuente_video}")
            return

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    # Si finaliza el video, reiniciar al inicio para bucle
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Fin del video.")
                        break

                frame = cv2.rotate(frame, self.ROTACION_CAMARA)
                panel = self._procesar_frame(frame)
                cv2.imshow(self.interfaz.nombre_ventana, panel)

                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
        finally:
            self.robot.cerrar()
            self.cap.release()
            cv2.destroyAllWindows()

    def _procesar_frame(self, frame):
        # 1. Línea: acción propuesta por el seguidor.
        salida, accion_linea = self.seguidor_linea.procesar(
            frame, self.interfaz.leer_controles_linea())

        # 2. Señal: la CNN decide si hay que parar o cambiar la velocidad.
        self._consultar_cnn(frame)

        # 3. Decisión final y envío al robot.
        accion = Accion.PARAR if self.control_cnn.detenido else accion_linea
        self.robot.intervalo_adelante = self.control_cnn.intervalo_adelante
        self.robot.enviar(accion)

        self._dibujar_estado(salida, accion)
        return self.visualizador.crear_panel(salida)

    def _consultar_cnn(self, frame):
        ahora = time.time()
        if ahora - self.ultima_prediccion < self.INTERVALO_PREDICCION_S:
            return
        self.ultima_prediccion = ahora
        clase, prob = self.predictor.predecir(frame)
        self.texto_cnn = f"CNN: {clase} {prob * 100:.0f}%"
        nueva = self.control_cnn.actualizar(clase, prob, ahora)
        if nueva:
            print(nueva)  # única salida por terminal: la señal detectada

    def _dibujar_estado(self, salida, accion):
        cv2.putText(salida, self.texto_cnn, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(salida, self.texto_cnn, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2, cv2.LINE_AA)

        detenido = self.control_cnn.detenido
        texto = f"{'DETENIDO' if detenido else 'SIGUIENDO'} -> {accion}"
        y = salida.shape[0] - 20
        cv2.putText(salida, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(salida, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    self.COLOR_DETENIDO if detenido else self.COLOR_SIGUIENDO, 2, cv2.LINE_AA)


if __name__ == "__main__":
    import sys
    # Uso: python main.py [fuente] [--prueba]
    #   python main.py 2                -> cámara 2, conectado al robot
    #   python main.py 2 --prueba       -> cámara 2, sin robot
    argumentos = [a for a in sys.argv[1:] if not a.startswith("--")]
    sin_robot = "--prueba" in sys.argv
    ruta_video = argumentos[0] if argumentos else "0"
    AplicacionDetector(fuente_video=ruta_video, sin_robot=sin_robot).ejecutar()
