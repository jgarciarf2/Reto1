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
#MAC_BLUETOOTH = "00:1B:10:31:0A:AB"
MAC_BLUETOOTH = "00:1B:10:21:2C:1B"
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
# main.py usa el mejor modelo que indica <carpeta>/seleccion.json (lo deja
# entrenar.py). Se usa la primera carpeta de esta lista que tenga el modelo
# (<carpeta>/<Modelo>/<Modelo>.keras): los .keras no se suben a git, así que
# cada computador usa el que haya entrenado o copiado.
CARPETAS_MODELO = ["resultados/reto2_exp3", "resultados/reto2_exp2", "resultados", "resultados/reto2_exp1"]
# Modelo que usa el robot. None = el primero de seleccion.json (menor val_loss).
# En reto2_exp2 ResNet50 gana por val_loss, pero MobileNet es mejor en test
# (92.5 % vs 83 %) y responde en menos de la mitad del tiempo (93 vs 214 ms).
MODELO_CNN = "MobileNet"

# Velocidades de marcha. El robot arranca en NORMAL.
#   ZONA_ESCOLAR  -> REDUCIDA
#   VELOCIDAD_60  -> NORMAL (sube si iba reducida; si ya iba normal, la conserva)
# `intervalo_s`: cada cuántos segundos se repite "adelante" (más espaciado =
# más lento). `avance_s`: si no es None, en cada intervalo el robot avanza
# esos segundos y luego se le manda "parar"; así frena aunque el mBot siga
# de largo con un solo "adelante". Los giros no se tocan.
VELOCIDADES = {
    "NORMAL": {"intervalo_s": 0.2, "avance_s": None},
    "REDUCIDA": {"intervalo_s": 1.0, "avance_s": 0.35},
}

# Cada cuántos segundos se consulta la CNN (es la parte pesada del loop).
INTERVALO_CNN_S = 0.2

# Giro del frame de la cámara en grados: 0, 90, -90 o 180. Tiene que quedar el
# robot abajo y la pista hacia arriba. Con DroidCam (celular sobre el carro)
# es 90; con el iPhone por Cámara de continuidad (video_test.MOV) era -90.
# También se puede probar sin editar: python main.py <camara> --rotar=-90
ROTACION_CAMARA = 90

# La rotación de arriba es para el seguidor de línea. La CNN reconoce mejor
# con el frame tal como sale de la cámara (en el video de prueba: girado
# -90 se pierden ZONA y SIGA). Pon False para pasarle el frame girado.
CNN_SIN_ROTAR = True
# Rellenar el frame hasta 3:4 / 4:3 antes de la CNN (como las fotos de
# entrenamiento). En el video de prueba reconoce mejor sin relleno.
CNN_RELLENO_3_4 = False

# Probabilidad mínima (0 a 1, promediada) para que la CNN cuente una señal.
# Más baja = reconoce más fácil pero se equivoca más. Mira el % en pantalla
# para ajustarla (o prueba con --confianza=0.7, que vale para todas).
# PARE y SIGA piden más: un PARE falso deja al robot quieto y un SIGA falso
# lo saca de un PARE (con la pista vacía la CNN tiende a decir SIGA con
# 40-85 %). ZONA y 60 solo cambian la velocidad, por eso van más bajos.
# Ajustados para el MobileNet de reto2_exp3 con el video de prueba: SIGA
# entre 0.84 y 0.88 nunca saca al robot de un PARE por error; con 0.90 ya
# no reconoce el SIGA real del final.
CONFIANZA_MINIMA = {
    "PARE": 0.90,
    "SIGA": 0.88,
    "ZONA_ESCOLAR": 0.50,
    "VELOCIDAD_60": 0.60,
    "SIN_SENAL": 0.60,
}
# Cuántas predicciones se promedian (suaviza frames sueltos con baja confianza).
VENTANA_CNN = 1
# Predicciones seguidas de la misma señal antes de obedecerla (cada 0.2 s:
# con 2 el robot reacciona en ~0.4 s).
CONFIRMACIONES_CNN = 2
