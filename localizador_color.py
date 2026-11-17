import cv2
import numpy as np

# ============================================================
# Localiza DÓNDE puede haber una señal en una imagen (no dice cuál es).
#
# Las señales son tarjetas con un color vivo (rojo: PARE y 60; verde: SIGA;
# amarillo: ZONA ESCOLAR) y la forma es compacta. Se segmenta ese color en
# HSV, se cierran huecos con morfología y se quedan los contornos compactos
# de buen tamaño. La CNN clasifica después SOLO el recorte, así no aprende
# del robot, la línea ni el fondo.
#
# Lo usan recortar_senales.py (para armar el dataset) y main.py (en vivo).
# ============================================================

# Rangos HSV de OpenCV (H 0-179). El rojo está en los dos extremos del matiz.
RANGOS = {
    "rojo": [((0, 110, 70), (6, 255, 255)), ((168, 110, 70), (179, 255, 255))],
    "verde": [((60, 40, 30), (98, 255, 255))],
    "amarillo": [((19, 110, 120), (27, 255, 255))],
}
# Segunda pasada para señales pálidas (poca luz o sobreexpuestas): menos saturación exigida.
RANGOS_RELAJADOS = {
    "rojo": [((0, 60, 60), (7, 255, 255)), ((165, 60, 60), (179, 255, 255))],
    "verde": [((60, 25, 25), (98, 255, 255))],
    # Segundo rango: la ZONA ESCOLAR sale casi blanca amarillenta en el video del iPhone
    # (H~30, S~65, V~235); las pilas amarillas son más oscuras (V~75), así que no entran.
    "amarillo": [((18, 60, 100), (28, 255, 255)), ((20, 40, 170), (34, 255, 255))],
}
# Color de cada clase del dataset (carpetas 0, 1, 2, 3)
COLOR_DE_CLASE = {"0": "rojo", "1": "verde", "2": "rojo", "3": "amarillo"}

LADO_DETECCION = 640        # se busca en una versión reducida (rápido)
AREA_MIN_FRAC = 0.003       # área mínima del contorno, como fracción de la imagen
SOLIDEZ_MIN = 0.75          # área / área de su envolvente convexa
ASPECTO = (0.45, 2.2)       # ancho / alto del recuadro
# Una señal tiene letras o pictograma: el color no llena todo el recuadro. Una pila
# amarilla o verde del robot sí (>90 %).
RELLENO = (0.15, 0.85)      # fracción del recuadro con el color
# Las pilas del robot tienen un degradado naranja -> amarillo -> verde. Si más de esta
# fracción del entorno del recuadro (ampliado un 40 %) es naranja o verde amarillento,
# el candidato es una pila, no una señal.
MAX_DEGRADADO = 0.10
# Una señal tiene partes blancas o negras (letras, pictograma, interior del 60). La mesa
# de madera o una pila no. Fracción mínima del recuadro con píxeles sin color.
NEUTRO_MIN = 0.10


def candidatos(frame, colores=("rojo", "verde", "amarillo"), relajado=False):
    """
    Devuelve las posibles señales como [{"caja": (x, y, w, h), "area": a, "color": c}],
    de mayor a menor área. Las cajas están en coordenadas de `frame`. Con
    `relajado=True` usa umbrales de color más permisivos (señales pálidas).
    """
    alto, ancho = frame.shape[:2]
    escala = min(1.0, LADO_DETECCION / max(alto, ancho))
    peq = cv2.resize(frame, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.GaussianBlur(peq, (5, 5), 0), cv2.COLOR_BGR2HSV)
    area_imagen = peq.shape[0] * peq.shape[1]
    k = max(3, int(0.012 * max(peq.shape[:2]))) | 1
    kernel = np.ones((k, k), np.uint8)

    h_all = hsv[:, :, 0]
    vivos = (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 80)
    bateria = vivos & (((h_all >= 6) & (h_all <= 15)) | ((h_all >= 32) & (h_all <= 55)))

    resultado = []
    for color in colores:
        mascara = np.zeros(peq.shape[:2], np.uint8)
        for bajo, alto_hsv in (RANGOS_RELAJADOS if relajado else RANGOS)[color]:
            mascara |= cv2.inRange(hsv, bajo, alto_hsv)
        cruda = mascara
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            area = cv2.contourArea(c)
            if area < AREA_MIN_FRAC * area_imagen:
                continue
            area_hull = cv2.contourArea(cv2.convexHull(c))
            if area_hull == 0 or area / area_hull < (SOLIDEZ_MIN - 0.08 if relajado else SOLIDEZ_MIN):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not ASPECTO[0] <= w / h <= ASPECTO[1]:
                continue
            relleno = float(np.count_nonzero(cruda[y:y + h, x:x + w])) / (w * h)
            if not RELLENO[0] <= relleno <= RELLENO[1]:
                continue
            roi = hsv[y:y + h, x:x + w]
            neutro = (roi[:, :, 1] < 60) | (roi[:, :, 2] < 60)
            if float(np.count_nonzero(neutro)) / (w * h) < NEUTRO_MIN:
                continue
            # Pila del robot: alrededor de su banda amarilla hay naranja y verde amarillento
            mx, my = int(w * 0.4), int(h * 0.4)
            x0, y0 = max(x - mx, 0), max(y - my, 0)
            x1, y1 = min(x + w + mx, hsv.shape[1]), min(y + h + my, hsv.shape[0])
            if float(np.count_nonzero(bateria[y0:y1, x0:x1])) / ((x1 - x0) * (y1 - y0)) > MAX_DEGRADADO:
                continue
            resultado.append({
                "caja": (int(x / escala), int(y / escala), int(w / escala), int(h / escala)),
                "area": area / escala ** 2,
                "color": color,
            })
    return sorted(resultado, key=lambda r: r["area"], reverse=True)


def recorte_cuadrado(frame, caja, margen=1.35, desplazamiento=(0.0, 0.0), lado_salida=256):
    """
    Recorta un cuadrado centrado en `caja` (lado = lado mayor * margen) y lo deja a
    lado_salida x lado_salida SIN deformar. `desplazamiento` mueve el centro, como
    fracción del lado, para hacer variantes. Si el cuadrado se sale de la imagen,
    se rellena repitiendo el borde.
    """
    x, y, w, h = caja
    lado = max(int(max(w, h) * margen), 16)
    cx = x + w / 2 + desplazamiento[0] * lado
    cy = y + h / 2 + desplazamiento[1] * lado
    x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    alto, ancho = frame.shape[:2]
    pad = max(0, -x0, -y0, x0 + lado - ancho, y0 + lado - alto)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + pad, y0 + pad
    recorte = frame[y0:y0 + lado, x0:x0 + lado]
    return cv2.resize(recorte, (lado_salida, lado_salida), interpolation=cv2.INTER_AREA)
