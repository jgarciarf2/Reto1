# ============================================================
# CONFIGURACIÓN DEL ROBOT
# ============================================================
# Este es el ÚNICO archivo que hay que tocar cuando en clase
# digan qué letra entiende el robot para cada acción, o por
# qué puerto se conecta. El resto del programa trabaja con los
# nombres de las acciones ("ADELANTE", "IZQUIERDA", ...) y
# nunca con las letras directamente.

# Letra que se envía al robot por cada acción según especificación del mBot:
# 'w' -> adelante
# 's' -> atras
# 'a' -> izquierda
# 'd' -> derecha
# 'x' -> parar
LETRAS = {
    "ADELANTE": "w",
    "IZQUIERDA": "a",
    "DERECHA": "d",
    "ATRAS": "s",
    "PARAR": "x",
}

# Dirección MAC de Bluetooth del mBot (dejar en None o vacío "" para modo pruebas/consola)
# Cambia esta MAC por la de tu robot asignado en clase:
MAC_BLUETOOTH = "00:1B:10:31:0A:AB"
PUERTO_BLUETOOTH = 1

# Si se usa puerto serial USB en lugar de socket Bluetooth directo:
PUERTO_SERIAL = None
BAUDIOS = 9600

TERMINADOR = ""

# Intervalo en segundos para enviar señales constantemente al robot (ej: cada 0.1s = 10 veces por seg)
INTERVALO_REFRESCO_S = 0.3


# ------------------------------------------------------------
# RETO 2: señales por CNN
# ------------------------------------------------------------
# Experimento entrenado (carpeta resultados/<EXPERIMENTO_CNN>/). Si no existe
# todavía, el programa usa el detector por color del reto 1.
EXPERIMENTO_CNN = "reto2_exp1"

# Velocidades de marcha. El mBot solo entiende "adelante" y "parar", así
# que la velocidad reducida se logra con pulsos: en cada ciclo de
# `ciclo_s` segundos avanza `avance_s` segundos y el resto queda quieto.
# NORMAL no tiene ciclo: "adelante" se repite cada INTERVALO_REFRESCO_S.
VELOCIDADES = {
    "NORMAL": {"ciclo_s": None, "avance_s": None},
    "REDUCIDA": {"ciclo_s": 1.0, "avance_s": 0.35},
}

# Probabilidad mínima (promediada) para creerle a la CNN, por señal.
# PARE y SIGA piden más: un PARE falso deja al robot quieto y un SIGA
# falso lo saca de un PARE. Con la pista vacía la CNN suele decir SIGA
# con 40-80 %, así que SIGA no puede bajar de ~0.90. ZONA y 60 solo
# cambian la velocidad y se ven con menos confianza en el video, por eso
# van más bajos. Ajustados con probar_videos.py sobre el video de prueba.
CONFIANZA_MINIMA = {
    "PARE": 0.90,
    "SIGA": 0.90,
    "ZONA_ESCOLAR": 0.60,
    "VELOCIDAD_60": 0.75,
}
# Cuántas predicciones se promedian (suaviza frames sueltos con baja confianza).
VENTANA_CNN = 2
# Predicciones seguidas de la misma señal antes de obedecerla.
CONFIRMACIONES_CNN = 3
# Cada cuántos segundos se consulta la CNN (es la parte pesada del loop).
INTERVALO_CNN_S = 0.2
