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
# Experimento entrenado (carpeta resultados/<EXPERIMENTO_CNN>/). main.py usa el
# mejor modelo de su seleccion.json.
EXPERIMENTO_CNN = "reto2_exp2"

# Cada cuántos segundos se envía "adelante" según la última señal vista:
# más espaciado = el robot avanza más despacio.
INTERVALOS_ADELANTE = {
    "SIGA": 1.0,
    "ZONA_ESCOLAR": 2.0,
    "VELOCIDAD_60": 0.5,
}

# Giro del frame de la cámara en grados: 0, 90, -90 o 180. Con el iPhone sobre
# el carro (video_test.MOV) -90 deja el robot abajo y la pista hacia arriba,
# igual que las fotos de entrenamiento. La webcam del Mac iría en 0.
# También se puede probar sin editar: python main.py 2 --rotar=90
ROTACION_CAMARA = -90

# Probabilidad mínima (0 a 1) para que la CNN cuente una señal. Más baja =
# reconoce más fácil pero se equivoca más. Mira el % en pantalla para ajustarla.
CONFIANZA_MINIMA = 0.85
