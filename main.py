import cv2

from interfaz_control import InterfazControl
from preprocesador import Preprocesador
from clasificador_color import ClasificadorColor
from detector_senales import DetectorSenales
from seguidor_linea import SeguidorLinea
from control_senales import ControlSenales
from comunicacion_robot import ComunicacionRobot, SalidaConsola
from visualizador import Visualizador


# ============================================================
# IDEA GENERAL
# ============================================================
# En cada frame de la cámara el programa hace dos cosas en
# paralelo y al final decide una sola acción para el robot:
#
#   Preprocesador     -> máscara de píxeles rojo/verde saturados
#                         (ROI, blur, HSV, umbral, morfología).
#   DetectorSenales   -> sobre esa máscara busca contornos con
#                         forma de octágono o de cuadrado girado
#                         (área, approxPolyDP, relación de
#                         aspecto, circularidad).
#   ClasificadorColor -> K-Means sobre el interior de cada
#                         figura para decidir PARE (rojo) o
#                         SIGA (verde).
#   SeguidorLinea     -> mide qué tan corrida está la línea negra
#                         respecto al centro del robot y propone
#                         ADELANTE, IZQUIERDA o DERECHA.
#   ControlSenales    -> máquina de estados: si vio PARE manda
#                         PARAR mientras la tarjeta siga ahí (o
#                         hasta ver SIGA); si no, deja pasar la
#                         acción del seguidor.
#   ComunicacionRobot -> traduce la acción a la letra que
#                         entiende el robot (config_robot.py) y
#                         la envía.
#   Visualizador      -> arma el panel que se ve en pantalla.
#   InterfazControl   -> ventanas + trackbars.
#
# Esta clase (AplicacionDetector) es la que arma todas las
# piezas y corre el loop de la cámara.


class AplicacionDetector:
    """
    Punto de entrada del programa: abre el archivo de video o cámara,
    arma cada pieza del pipeline y corre el loop principal (leer frame,
    procesarlo, mostrarlo) hasta que el usuario presione 'q'.
    """

    COLORES_ESTADO = {
        ControlSenales.SIGUIENDO: (0, 200, 0),
        ControlSenales.DETENIDO: (0, 0, 255),
    }
    # El stream del movil llega apaisado aunque el telefono este vertical.
    # Cambia a ROTATE_90_COUNTERCLOCKWISE si la imagen queda invertida.
    ROTACION_CAMARA = cv2.ROTATE_90_CLOCKWISE

    def __init__(self, fuente_video="0", sin_robot=False):
        self.fuente_video = fuente_video
        # Un número ("0", "1", ...) o int es el índice de la cámara física;
        # cualquier otra cosa es una ruta a un archivo de video.
        if isinstance(fuente_video, str) and fuente_video.isdigit():
            indice = int(fuente_video)
        elif isinstance(fuente_video, int):
            indice = fuente_video
        else:
            indice = fuente_video

        # Primero el robot: conectarse por Bluetooth puede tardar
        # varios segundos, y si las ventanas ya estuvieran abiertas
        # se quedarían grises y congeladas mientras tanto.
        # En modo prueba no se conecta: las letras solo se imprimen.
        if sin_robot:
            print("Modo prueba: sin robot, las letras solo se imprimen en consola.")
            self.robot = ComunicacionRobot(salida=SalidaConsola())
        else:
            self.robot = ComunicacionRobot()

        self.cap = cv2.VideoCapture(indice)

        self.interfaz = InterfazControl()
        self.preprocesador = Preprocesador()
        self.clasificador_color = ClasificadorColor()
        self.detector = DetectorSenales(self.clasificador_color)
        self.seguidor_linea = SeguidorLinea()
        self.control = ControlSenales()
        self.visualizador = Visualizador()

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
        parametros_senales = self.interfaz.leer_controles_senales()
        parametros_linea = self.interfaz.leer_controles_linea()

        # 1. Señales: máscaras de color -> contornos -> formas.
        mascaras = self.preprocesador.mascaras_senales(frame, parametros_senales)
        contornos = self.preprocesador.encontrar_contornos(mascaras["total"])
        salida, detecciones = self.detector.procesar(frame, contornos, parametros_senales)
        area_roja = self.preprocesador.area_mayor(mascaras["roja"])

        # 2. Línea: acción propuesta por el seguidor.
        salida, accion_linea = self.seguidor_linea.procesar(frame, parametros_linea, salida)

        # 3. Decisión final y envío al robot.
        accion = self.control.actualizar(detecciones, area_roja, accion_linea, parametros_senales)
        self.robot.enviar(accion)

        self._dibujar_estado(salida, accion)
        return self.visualizador.crear_panel(salida, mascaras["total"])

    def _dibujar_estado(self, salida, accion):
        estado = self.control.estado
        texto = f"{estado} -> {accion}"
        y = salida.shape[0] - 20
        cv2.putText(salida, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(salida, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    self.COLORES_ESTADO[estado], 2, cv2.LINE_AA)


if __name__ == "__main__":
    import sys
    # Uso: python main.py [fuente] [--prueba]
    #   python main.py 2                -> cámara 2, conectado al robot
    #   python main.py 2 --prueba       -> cámara 2, sin robot (solo consola)
    argumentos = [a for a in sys.argv[1:] if not a.startswith("--")]
    sin_robot = "--prueba" in sys.argv
    ruta_video = argumentos[0] if argumentos else "0"
    AplicacionDetector(fuente_video=ruta_video, sin_robot=sin_robot).ejecutar()
