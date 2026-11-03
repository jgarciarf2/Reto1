import unittest
from itertools import groupby
from unittest import mock

from comunicacion_robot import Accion, ComunicacionRobot
from control_cnn import ControlCNN


class SalidaMemoria:
    """Guarda las acciones enviadas en vez de mandarlas al robot."""

    def __init__(self):
        self.acciones = []

    def escribir(self, texto, accion):
        self.acciones.append(accion)

    def cerrar(self):
        pass


CLASES = ["PARE", "SIGA", "VELOCIDAD_60", "ZONA_ESCOLAR"]


def probs(clase, prob):
    """Probabilidades de un frame: `prob` para `clase` y el resto repartido."""
    resto = (1 - prob) / (len(CLASES) - 1)
    return {c: prob if c == clase else resto for c in CLASES}


def ver(control, clase, veces=None, prob=0.99):
    # Al cambiar de señal el promedio tarda (ventana - 1) frames en "olvidar" la anterior
    veces = veces or control.confirmaciones + control.ventana - 1
    for _ in range(veces):
        control.actualizar(probs(clase, prob))


class TestControlCNN(unittest.TestCase):

    def setUp(self):
        self.c = ControlCNN(confirmaciones=3, ventana=2)

    def test_arranca_en_marcha_a_velocidad_normal(self):
        self.assertFalse(self.c.detenido)
        self.assertEqual(self.c.velocidad, ControlCNN.NORMAL)

    def test_pare_detiene_y_sigue_detenido_sin_siga(self):
        ver(self.c, "PARE")
        self.assertTrue(self.c.detenido)
        # Aunque se quite la tarjeta (la CNN ya no ve PARE) sigue quieto
        ver(self.c, "VELOCIDAD_60", veces=10, prob=0.40)
        self.assertTrue(self.c.detenido)

    def test_siga_reanuda_la_marcha(self):
        ver(self.c, "PARE")
        ver(self.c, "SIGA")
        self.assertFalse(self.c.detenido)

    def test_zona_y_60_no_sacan_de_un_pare(self):
        ver(self.c, "PARE")
        ver(self.c, "ZONA_ESCOLAR")
        self.assertTrue(self.c.detenido)
        ver(self.c, "VELOCIDAD_60")
        self.assertTrue(self.c.detenido)

    def test_zona_reduce_y_60_recupera_la_normal(self):
        ver(self.c, "ZONA_ESCOLAR")
        self.assertEqual(self.c.velocidad, ControlCNN.REDUCIDA)
        ver(self.c, "VELOCIDAD_60")
        self.assertEqual(self.c.velocidad, ControlCNN.NORMAL)

    def test_60_a_velocidad_normal_la_conserva(self):
        ver(self.c, "VELOCIDAD_60")
        self.assertEqual(self.c.velocidad, ControlCNN.NORMAL)

    def test_siga_en_marcha_no_cambia_la_velocidad(self):
        ver(self.c, "ZONA_ESCOLAR")
        ver(self.c, "SIGA")
        self.assertFalse(self.c.detenido)
        self.assertEqual(self.c.velocidad, ControlCNN.REDUCIDA)

    def test_zona_vista_detenido_se_aplica_al_arrancar(self):
        ver(self.c, "PARE")
        ver(self.c, "ZONA_ESCOLAR")
        ver(self.c, "SIGA")
        self.assertFalse(self.c.detenido)
        self.assertEqual(self.c.velocidad, ControlCNN.REDUCIDA)

    def test_pocas_confirmaciones_no_cuentan(self):
        ver(self.c, "PARE", veces=2)
        self.assertFalse(self.c.detenido)

    def test_baja_confianza_no_cuenta_y_corta_la_racha(self):
        for p in (0.99, 0.99, 0.30, 0.99):
            self.c.actualizar(probs("PARE", p))
        self.assertFalse(self.c.detenido)

    def test_el_promedio_tolera_un_frame_algo_bajo(self):
        # 0.80 solo no pasa el umbral de ZONA en promedio con los vecinos altos sí
        for p in (0.95, 0.80, 0.95, 0.95):
            self.c.actualizar(probs("ZONA_ESCOLAR", p))
        self.assertEqual(self.c.velocidad, ControlCNN.REDUCIDA)

    def test_pista_vacia_con_siga_dudoso_no_saca_de_un_pare(self):
        ver(self.c, "PARE")
        # Lo típico con la pista vacía: SIGA entre 40 y 80 %
        for p in (0.6, 0.8, 0.75, 0.7, 0.8, 0.65):
            self.c.actualizar(probs("SIGA", p))
        self.assertTrue(self.c.detenido)


class TestVelocidadRobot(unittest.TestCase):

    def simular(self, velocidad, accion=Accion.ADELANTE, segundos=2.0, fps=20):
        salida = SalidaMemoria()
        robot = ComunicacionRobot(salida=salida)
        robot.velocidad = velocidad
        tiempos = []
        for k in range(int(segundos * fps)):
            t = 100 + k / fps
            antes = len(salida.acciones)
            with mock.patch("comunicacion_robot.time.time", return_value=t):
                robot.enviar(accion)
            tiempos += [(round(t - 100, 2), a) for a in salida.acciones[antes:]]
        return tiempos

    def test_normal_repite_adelante_sin_parar(self):
        envios = self.simular("NORMAL")
        self.assertEqual({a for _, a in envios}, {Accion.ADELANTE})
        # 2 s con "adelante" cada 0.4 s
        self.assertEqual(len(envios), 5)

    def test_reducida_avanza_y_para_en_cada_intervalo(self):
        envios = self.simular("REDUCIDA")
        cambios = [a for a, _ in groupby(a for _, a in envios)]
        self.assertEqual(cambios, [Accion.ADELANTE, Accion.PARAR,
                                   Accion.ADELANTE, Accion.PARAR])
        # el "parar" llega a los 0.35 s del "adelante" (con un frame de margen)
        t_parar, accion = envios[1]
        self.assertEqual(accion, Accion.PARAR)
        self.assertTrue(0.35 <= t_parar <= 0.40, t_parar)

    def test_reducida_manda_menos_adelante_que_normal(self):
        def adelantes(v):
            return sum(a == Accion.ADELANTE for _, a in self.simular(v, segundos=4))
        self.assertLess(adelantes("REDUCIDA"), adelantes("NORMAL"))

    def test_los_giros_no_se_tocan(self):
        envios = self.simular("REDUCIDA", accion=Accion.IZQUIERDA, segundos=1)
        self.assertEqual({a for _, a in envios}, {Accion.IZQUIERDA})


if __name__ == "__main__":
    unittest.main()
