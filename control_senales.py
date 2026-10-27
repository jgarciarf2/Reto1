from comunicacion_robot import Accion


class ControlSenales:
    """
    Máquina de estados que decide si el robot obedece al
    seguidor de línea o se queda quieto por una señal.

        SIGUIENDO --(PARE confirmado)----------------> DETENIDO
        DETENIDO  --(SIGA confirmado)----------------> SIGUIENDO
        DETENIDO  --(la tarjeta roja desaparece)------> SIGUIENDO

    - En SIGUIENDO se envía la acción del seguidor de línea.
    - En DETENIDO se envía PARAR, sin importar la línea.
    - Para frenar, el PARE tiene que verse con forma válida
      varios frames seguidos (evita que un falso positivo de un
      solo frame frene al robot).
    - Para seguir detenido basta con que la mancha roja siga a
      la vista, aunque la mano del profesor tape parte de la
      tarjeta y ya no se vea la forma completa (histéresis: es
      más difícil entrar al estado que quedarse en él).
    - Cuando la tarjeta lleva varios frames sin verse (la
      quitaron), o cuando aparece SIGA, el robot sigue.
    """

    SIGUIENDO = "SIGUIENDO"
    DETENIDO = "DETENIDO"

    def __init__(self):
        self.estado = self.SIGUIENDO
        self.senal_anterior = None
        self.frames_senal = 0
        self.frames_sin_pare = 0

    def actualizar(self, detecciones, area_roja, accion_linea, parametros):
        """
        Recibe las detecciones del frame (ver
        DetectorSenales.procesar), el área de la mancha roja más
        grande, la acción que propone el seguidor de línea y los
        parámetros de los trackbars. Devuelve la acción que hay
        que enviarle al robot.
        """
        senal = self._senal_confirmada(detecciones, parametros["frames_confirmacion"])

        if self.estado == self.SIGUIENDO:
            if senal == "PARE":
                self.estado = self.DETENIDO
                self.frames_sin_pare = 0
        elif senal == "SIGA":
            self.estado = self.SIGUIENDO
        else:
            # Se considera que la tarjeta sigue ahí si hay un PARE
            # con forma válida o, al menos, una mancha roja de
            # buen tamaño (la mitad del área mínima de señal).
            pare_visible = self.senal_anterior == "PARE" \
                or area_roja >= parametros["area_minima"] / 2
            self.frames_sin_pare = 0 if pare_visible else self.frames_sin_pare + 1

            if self.frames_sin_pare >= parametros["frames_liberar"]:
                self.estado = self.SIGUIENDO

        if self.estado == self.DETENIDO:
            return Accion.PARAR
        return accion_linea

    def _senal_confirmada(self, detecciones, frames_confirmacion):
        # Si en el frame hay varias señales, manda la más grande
        # (la más cercana al robot). Las de color dudoso no
        # cuentan.
        validas = [d for d in detecciones if d["etiqueta"] in ("PARE", "SIGA")]
        senal = max(validas, key=lambda d: d["area"])["etiqueta"] if validas else None

        if senal is not None and senal == self.senal_anterior:
            self.frames_senal += 1
        else:
            self.frames_senal = 1 if senal is not None else 0
        self.senal_anterior = senal

        if senal is not None and self.frames_senal >= frames_confirmacion:
            return senal
        return None
