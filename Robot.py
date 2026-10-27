import socket
import time


class Robot:
    """
    Control de robot mBot mediante conexión Bluetooth RFCOMM (socket).
    Comandos enviados por Bluetooth:
      'w' -> adelante
      's' -> atras
      'a' -> izquierda
      'd' -> derecha
      'x' -> parar
    """
    def __init__(self, mac_address: str, port: int = 1):
        self.mac_address = mac_address
        self.port = port
        self.bluetooth_socket = None

    def conectar(self):
        if not self.mac_address:
            print("[Robot] No se configuró MAC Bluetooth. Modo simulación activo.")
            return

        self.bluetooth_socket = socket.socket(
            socket.AF_BLUETOOTH,
            socket.SOCK_STREAM,
            socket.BTPROTO_RFCOMM
        )

        try:
            print(f"Conectando a {self.mac_address}...")
            self.bluetooth_socket.connect((self.mac_address, self.port))
            print("Conexión establecida")
        except OSError as e:
            print(f"Error al conectar por Bluetooth: {e}")
            self.cerrar()
            raise

    def _enviar(self, comando: str):
        if self.bluetooth_socket is None:
            # Si no hay socket conectado, solo imprimimos
            return

        try:
            self.bluetooth_socket.sendall(comando.encode("utf-8"))
            print(f"Comando enviado: {comando}")
        except OSError as e:
            print(f"Error enviando comando: {e}")

    def adelante(self):
        self._enviar("w")

    def atras(self):
        self._enviar("s")

    def izquierda(self):
        self._enviar("a")

    def derecha(self):
        self._enviar("d")

    def parar(self):
        self._enviar("x")

    def cerrar(self):
        if self.bluetooth_socket is not None:
            try:
                self.bluetooth_socket.close()
            except Exception:
                pass
            self.bluetooth_socket = None
            print("Conexión cerrada")
