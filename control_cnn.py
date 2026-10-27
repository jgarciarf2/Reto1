import time

import config_robot


class ControlCNN:
    """
    Convierte las predicciones de la CNN en decisiones para el robot:

        PARE          -> detenido (se queda quieto mientras la CNN siga viendo PARE)
        SIGA          -> avanza a ritmo normal
        ZONA_ESCOLAR  -> avanza despacio
        VELOCIDAD_60  -> avanza rápido

    La velocidad se expresa como cada cuántos segundos se envía "adelante"
    (ver config_robot.INTERVALOS_ADELANTE): más espaciado = más lento.
    Una señal solo cuenta si se repite `confirmaciones` predicciones
    seguidas con confianza >= `confianza_minima`.
    """

    def __init__(self, confianza_minima=0.85, confirmaciones=3, segundos_liberar=1.5):
        self.confianza_minima = confianza_minima
        self.confirmaciones = confirmaciones
        self.segundos_liberar = segundos_liberar
        self.detenido = False
        self.senal_vigente = "SIGA"
        self._candidata = None
        self._repeticiones = 0
        self._ultimo_pare = 0.0
        self._ultima_confirmada = None
        self._ultimo_confiable = 0.0

    @property
    def intervalo_adelante(self):
        return config_robot.INTERVALOS_ADELANTE[self.senal_vigente]

    def actualizar(self, clase, prob, ahora=None):
        """Devuelve la señal recién confirmada (o None si no hay una nueva)."""
        ahora = ahora or time.time()
        nueva = None
        confiable = prob >= self.confianza_minima
        if confiable:
            self._ultimo_confiable = ahora
        elif ahora - self._ultimo_confiable > 1.0:
            # Un segundo sin ver ninguna señal: la próxima vuelve a contar como nueva
            self._ultima_confirmada = None

        if confiable and clase == self._candidata:
            self._repeticiones += 1
        else:
            self._candidata = clase if confiable else None
            self._repeticiones = 1 if confiable else 0

        if self._repeticiones >= self.confirmaciones:
            if clase != self._ultima_confirmada:
                nueva = self._ultima_confirmada = clase
            if clase == "PARE":
                self.detenido = True
                self._ultimo_pare = ahora
            else:
                self.detenido = False
                self.senal_vigente = clase

        # Si la tarjeta de PARE ya no se ve, el robot sigue
        if self.detenido and ahora - self._ultimo_pare > self.segundos_liberar:
            self.detenido = False
            self._ultima_confirmada = None
        return nueva
