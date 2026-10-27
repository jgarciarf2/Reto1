import time
import config_robot


class Accion:
    """
    Nombres de las acciones que el programa le puede pedir al
    robot. Son solo nombres: la letra real de cada una vive en
    config_robot.LETRAS.
    """
    ADELANTE = "ADELANTE"
    IZQUIERDA = "IZQUIERDA"
    DERECHA = "DERECHA"
    PARAR = "PARAR"


class SalidaConsola:
    """Canal de prueba: imprime la letra en vez de enviarla."""

    def escribir(self, texto, accion):
        print(f"[robot] {accion:<9} -> {texto!r}")

    def cerrar(self):
        pass


class SalidaSerial:
    """
    Canal real: envía la letra por el puerto serial (cable USB
    o módulo Bluetooth emparejado como puerto COM). Necesita
    la librería pyserial (pip install pyserial).
    """

    def __init__(self, puerto, baudios):
        import serial
        self.conexion = serial.Serial(puerto, baudios, timeout=1)

    def escribir(self, texto, accion):
        self.conexion.write(texto.encode("ascii"))

    def cerrar(self):
        self.conexion.close()


class SalidaBluetooth:
    """
    Canal Bluetooth directo mediante RFCOMM socket (Robot.py del profesor).
    """

    def __init__(self, mac_address, port=1):
        from Robot import Robot
        self.robot = Robot(mac_address, port)
        try:
            self.robot.conectar()
        except Exception as e:
            print(f"[SalidaBluetooth] No se pudo conectar a {mac_address}: {e}")
            print("[SalidaBluetooth] Pasando a modo consola.")
            self.robot = None

    def escribir(self, texto, accion):
        if self.robot is not None:
            # Eliminar posibles terminadores para enviar el comando limpio
            comando = texto.strip()
            self.robot._enviar(comando)
        print(f"[robot] {accion:<9} -> {texto!r}")

    def cerrar(self):
        if self.robot is not None:
            self.robot.parar()
            self.robot.cerrar()


class ComunicacionRobot:
    """
    Traduce una Accion a su letra (según config_robot) y la
    envía por el canal configurado.
    Envía constantemente señales activas (adelante, izquierda, etc.)
    cada INTERVALO_REFRESCO_S (o inmediatamente si la acción cambia)
    para mantener el robot en movimiento fluido.
    """

    def __init__(self, letras=None, salida=None):
        self.letras = letras or config_robot.LETRAS
        self.salida = salida or self._crear_salida()
        self.ultima_accion = None
        self.ultimo_tiempo_envio = 0
        self.intervalo = getattr(config_robot, "INTERVALO_REFRESCO_S", 0.3)

    def _crear_salida(self):
        if getattr(config_robot, "MAC_BLUETOOTH", None):
            return SalidaBluetooth(
                config_robot.MAC_BLUETOOTH,
                getattr(config_robot, "PUERTO_BLUETOOTH", 1)
            )
        if config_robot.PUERTO_SERIAL:
            return SalidaSerial(config_robot.PUERTO_SERIAL, config_robot.BAUDIOS)
        return SalidaConsola()

    def enviar(self, accion):
        ahora = time.time()
        # Enviar inmediatamente si cambió la acción, o periódicamente si sigue siendo la misma
        debe_enviar = (accion != self.ultima_accion) or ((ahora - self.ultimo_tiempo_envio) >= self.intervalo)

        if not debe_enviar:
            return

        texto = self.letras[accion] + config_robot.TERMINADOR
        self.salida.escribir(texto, accion)
        self.ultima_accion = accion
        self.ultimo_tiempo_envio = ahora

    def cerrar(self):
        # Antes de soltar la conexión se deja el robot quieto.
        texto = self.letras[Accion.PARAR] + config_robot.TERMINADOR
        self.salida.escribir(texto, Accion.PARAR)
        self.salida.cerrar()

