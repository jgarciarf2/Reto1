from collections import deque

import numpy as np

import config_robot


class ControlCNN:
    """
    Convierte las predicciones de la CNN en el estado del robot, según la
    tabla de comportamientos del reto 2:

        PARE          -> se detiene y se queda detenido HASTA ver SIGA
        SIGA          -> arranca si estaba detenido; si no, sigue igual
        ZONA_ESCOLAR  -> velocidad REDUCIDA (no saca al robot de un PARE)
        VELOCIDAD_60  -> velocidad NORMAL   (no saca al robot de un PARE)

    El estado tiene dos partes independientes: `detenido` (solo lo cambian
    PARE y SIGA) y `velocidad` (solo la cambian ZONA_ESCOLAR y
    VELOCIDAD_60). Así, si el robot está detenido y ve la zona escolar,
    sigue quieto, pero cuando le muestren SIGA arranca ya despacio.

    La CNN siempre responde una de las cuatro clases, también cuando la
    cámara solo ve la pista (y ahí tiende a decir SIGA con 40-80 %). Por
    eso una señal solo cuenta si:
      - el PROMEDIO de las últimas `ventana` predicciones la pone por
        encima del umbral de esa clase (config_robot.CONFIANZA_MINIMA).
        PARE y SIGA piden más porque equivocarse con ellas deja al robot
        quieto o lo saca de un PARE; ZONA y 60 solo cambian la velocidad.
      - se mantiene `confirmaciones` predicciones seguidas.
    """

    NORMAL = "NORMAL"
    REDUCIDA = "REDUCIDA"

    def __init__(self, confirmaciones=None, confianza_minima=None, ventana=None,
                 arranca_detenido=False):
        self.confirmaciones = confirmaciones or config_robot.CONFIRMACIONES_CNN
        self.confianza_minima = confianza_minima or config_robot.CONFIANZA_MINIMA
        self.ventana = ventana or config_robot.VENTANA_CNN
        self.detenido = arranca_detenido
        self.velocidad = self.NORMAL
        self.ultima_senal = None  # última señal confirmada (para mostrar en pantalla)
        self._historial = deque(maxlen=self.ventana)
        self._candidata = None
        self._repeticiones = 0

    def actualizar(self, probs):
        """
        Recibe las probabilidades de un frame ({clase: prob}) y devuelve la
        señal que se confirmó en este frame (o None).
        """
        clases = list(probs)
        self._historial.append([probs[c] for c in clases])
        promedio = np.mean(self._historial, axis=0)
        i = int(np.argmax(promedio))
        clase, prob = clases[i], float(promedio[i])

        if prob < self.confianza_minima.get(clase, 1.0):
            self._candidata, self._repeticiones = None, 0
            return None

        if clase == self._candidata:
            self._repeticiones += 1
        else:
            self._candidata, self._repeticiones = clase, 1

        # Solo se actúa una vez por aparición: al llegar justo a las confirmaciones
        if self._repeticiones != self.confirmaciones:
            return None

        self.ultima_senal = clase
        if clase == "PARE":
            self.detenido = True
        elif clase == "SIGA":
            self.detenido = False
        elif clase == "ZONA_ESCOLAR":
            self.velocidad = self.REDUCIDA
        elif clase == "VELOCIDAD_60":
            self.velocidad = self.NORMAL
        return clase
