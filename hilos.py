import threading
import time

import cv2


# ============================================================
# TRABAJO EN PARALELO
# ============================================================
# El loop principal solo sigue la línea y le habla al robot. Las dos
# cosas lentas corren aparte, cada una en su hilo (OpenCV y TensorFlow
# sueltan el GIL de Python mientras trabajan, así que los hilos sí
# corren a la vez en distintos núcleos del procesador):
#
#   CamaraEnVivo -> lee la cámara sin parar y guarda SOLO el último
#                   frame. Sin esto, OpenCV acumula en cola los frames
#                   que el loop no alcanza a procesar y el robot ve la
#                   pista con segundos de atraso (y zigzaguea).
#   HiloCNN      -> clasifica la señal del frame más reciente cada
#                   INTERVALO_CNN_S, sin frenar al seguidor de línea.


class CamaraEnVivo:
    """Lector de cámara en segundo plano: read() siempre da el frame más nuevo."""

    def __init__(self, cap):
        self.cap = cap
        self._frame = None
        self._numero = 0  # cuántos frames han llegado (para no repetir el mismo)
        self._leido = 0
        self._lock = threading.Lock()
        self._nuevo = threading.Condition(self._lock)
        self._activo = True
        self._hilo = threading.Thread(target=self._leer, daemon=True, name="camara")
        self._hilo.start()

    def _leer(self):
        fallos = 0
        while self._activo:
            ok, frame = self.cap.read()
            if not ok:
                fallos += 1
                if fallos > 50:  # ~5 s sin imágenes
                    break
                time.sleep(0.1)
                continue
            fallos = 0
            with self._nuevo:
                self._frame = frame
                self._numero += 1
                self._nuevo.notify_all()
        with self._nuevo:
            self._activo = False
            self._nuevo.notify_all()

    def read(self, espera_s=5.0):
        """Espera un frame que no se haya entregado todavía y lo devuelve (ok, frame)."""
        with self._nuevo:
            self._nuevo.wait_for(lambda: self._numero > self._leido or not self._activo,
                                 timeout=espera_s)
            if self._numero == self._leido:
                return False, None
            self._leido = self._numero
            return True, self._frame

    def release(self):
        self._activo = False
        self._hilo.join(timeout=1)
        self.cap.release()


class HiloCNN:
    """
    Corre la CNN en segundo plano sobre el último frame que se le pase
    con `ver(frame)`. Cada INTERVALO_CNN_S toma ese frame, calcula las
    probabilidades y actualiza el ControlCNN. El loop principal solo lee
    el resultado (`control.detenido`, `control.velocidad`, `texto`, ...).
    """

    def __init__(self, predictor, control, intervalo_s, al_confirmar=None):
        self.predictor = predictor
        self.control = control
        self.intervalo_s = intervalo_s
        self.al_confirmar = al_confirmar
        self.texto = "CNN..."
        self.lineas_probs = []
        self.ms_por_prediccion = 0.0
        self._frame = None
        self._lock = threading.Lock()
        self._activo = True
        self._hilo = threading.Thread(target=self._correr, daemon=True, name="cnn")
        self._hilo.start()

    def ver(self, frame):
        with self._lock:
            self._frame = frame

    def _correr(self):
        while self._activo:
            inicio = time.time()
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is not None:
                probs = self.predictor.probabilidades(frame)
                self.ms_por_prediccion = (time.time() - inicio) * 1000
                clase = max(probs, key=probs.get)
                self.texto = f"CNN: {clase} {probs[clase] * 100:.0f}%"
                self.lineas_probs = [f"{c} {p * 100:.0f}%" for c, p in probs.items()]
                nueva = self.control.actualizar(probs)
                if nueva and self.al_confirmar:
                    self.al_confirmar(nueva)
            # Mismo ritmo de consulta con el que se ajustaron los umbrales
            time.sleep(max(0.0, self.intervalo_s - (time.time() - inicio)))

    def detener(self):
        self._activo = False
        self._hilo.join(timeout=1)
