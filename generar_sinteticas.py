"""
Genera fotos sintéticas para equilibrar el dataset.

Problema: las fotos tomadas en la pista con el robot (pista_*.jpg) solo
tienen PARE y SIGA. La CNN aprendió el atajo "robot sobre la pista = PARE o
SIGA" y empezó a confundir la ZONA ESCOLAR con PARE.

Solución (sin tomar más fotos):
  1. Fondos: a las fotos pista_* se les borra la señal (inpainting), así
     queda la pista con el robot y sin ninguna señal.
  2. Recortes: de las fotos reales de cada clase se recorta la señal por
     color (rojo, verde, amarillo) y su envolvente convexa.
  3. Montaje: cada señal se pega sobre un fondo con posición, tamaño y
     giro al azar. Las cuatro señales quedan con la MISMA escena.

Las sintéticas se llaman sintetica_*.jpg y el entrenador las deja siempre
en train: validación y test siguen siendo solo fotos reales.

    python generar_sinteticas.py            # 28 fotos de pista por clase
    python generar_sinteticas.py 40

El número es cuántas fotos "de pista" (reales pista_* + sintéticas) debe
tener cada clase: a PARE y SIGA, que ya tienen fotos reales de la pista,
se les generan menos, para que ninguna clase quede asociada a la escena.
"""
import glob
import os
import random
import sys

import cv2
import numpy as np

DATASET_DIR = "5_cnns/dataset"
CARPETAS = {"PARE": "0", "SIGA": "1", "VELOCIDAD_60": "2", "ZONA_ESCOLAR": "3"}

# Rangos HSV (OpenCV: H 0-179) del color principal de cada señal
COLORES = {
    "PARE": [((0, 90, 70), (8, 255, 255)), ((168, 90, 70), (179, 255, 255))],
    "SIGA": [((45, 70, 40), (90, 255, 255))],
    "VELOCIDAD_60": [((0, 90, 70), (8, 255, 255)), ((168, 90, 70), (179, 255, 255))],
    "ZONA_ESCOLAR": [((18, 70, 110), (35, 255, 255))],
}


def mascara_color(img, clase):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = np.zeros(img.shape[:2], np.uint8)
    for bajo, alto in COLORES[clase]:
        m |= cv2.inRange(hsv, bajo, alto)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def senal_en(img, clase, zona=None):
    """
    Busca la señal de `clase` en la imagen: la mancha de color más grande
    (dentro de `zona` = (y0, y1) si se da) que sea compacta, y devuelve el
    polígono de su envolvente convexa (o None).
    """
    m = mascara_color(img, clase)
    if zona:
        m[:zona[0]] = 0
        m[zona[1]:] = 0
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alto, ancho = img.shape[:2]
    mejores = []
    for c in contornos:
        casco = cv2.convexHull(c)
        area = cv2.contourArea(casco)
        if area < 0.002 * alto * ancho:
            continue
        x, y, w, h = cv2.boundingRect(casco)
        if x <= 2 or y <= 2 or x + w >= ancho - 2 or y + h >= alto - 2:
            continue  # cortada por el borde
        if not 0.6 < w / h < 1.6:
            continue
        perimetro = cv2.arcLength(casco, True)
        if 4 * np.pi * area / perimetro ** 2 < 0.65:  # poco compacta: no es una señal
            continue
        mejores.append((area, casco))
    return max(mejores, key=lambda t: t[0])[1] if mejores else None


def fondos_vacios():
    """Fotos pista_* con la señal borrada por inpainting."""
    fondos = []
    for clase in ("PARE", "SIGA"):
        for ruta in sorted(glob.glob(os.path.join(DATASET_DIR, CARPETAS[clase], "pista_*.jpg"))):
            img = cv2.imread(ruta, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            casco = senal_en(img, clase, zona=(0, int(img.shape[0] * 0.65)))
            if casco is None:
                print(f"  (sin señal clara, se omite como fondo) {ruta}")
                continue
            mascara = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(mascara, [casco], 255)
            mascara = cv2.dilate(mascara, np.ones((25, 25), np.uint8))  # incluye el borde blanco
            fondos.append(cv2.inpaint(img, mascara, 9, cv2.INPAINT_TELEA))
    return fondos


def recortes(clase, maximo=15):
    """Señales recortadas (BGR + alfa) de las fotos reales de la clase."""
    salida = []
    rutas = sorted(glob.glob(os.path.join(DATASET_DIR, CARPETAS[clase], "*")))
    rutas = [r for r in rutas if not os.path.basename(r).startswith("sintetica_")]
    random.shuffle(rutas)
    for ruta in rutas:
        img = cv2.imread(ruta, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if img is None:
            continue
        escala = 1200 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        casco = senal_en(img, clase)
        if casco is None:
            continue
        x, y, w, h = cv2.boundingRect(casco)
        if w < 80:  # muy pequeña: el recorte quedaría borroso
            continue
        alfa = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(alfa, [casco], 255)
        alfa = cv2.dilate(alfa, np.ones((5, 5), np.uint8))
        recorte = np.dstack([img, alfa])[y:y + h, x:x + w]
        salida.append(recorte)
        if len(salida) >= maximo:
            break
    return salida


def pegar(fondo, recorte, rng):
    """Pega la señal girada y escalada en un punto al azar por encima del robot."""
    fondo = fondo.copy()
    alto, ancho = fondo.shape[:2]
    lado = int(ancho * rng.uniform(0.13, 0.22))  # tamaño típico de la señal en las fotos pista_*
    escala = lado / max(recorte.shape[:2])
    rec = cv2.resize(recorte, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    # Giro libre (las señales aparecen en cualquier ángulo)
    h, w = rec.shape[:2]
    diag = int(np.hypot(h, w)) + 2
    lienzo = np.zeros((diag, diag, 4), np.uint8)
    lienzo[(diag - h) // 2:(diag - h) // 2 + h, (diag - w) // 2:(diag - w) // 2 + w] = rec
    giro = cv2.getRotationMatrix2D((diag / 2, diag / 2), rng.uniform(0, 360), 1)
    rec = cv2.warpAffine(lienzo, giro, (diag, diag))

    # Posición: zona de la pista delante del robot (arriba del 60 % del alto)
    x = int(rng.uniform(0.05 * ancho, ancho - diag - 0.05 * ancho))
    y = int(rng.uniform(0.05 * alto, 0.60 * alto - diag))
    alfa = cv2.GaussianBlur(rec[:, :, 3], (5, 5), 0)[..., None] / 255.0
    zona = fondo[y:y + diag, x:x + diag].astype(np.float32)
    # Ajuste suave de brillo para que la señal no "brille" más que la escena
    color = rec[:, :, :3].astype(np.float32) * rng.uniform(0.85, 1.05)
    fondo[y:y + diag, x:x + diag] = np.clip(alfa * color + (1 - alfa) * zona, 0, 255).astype(np.uint8)
    return fondo


def main():
    objetivo = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    rng = random.Random(42)
    random.seed(42)

    for ruta in glob.glob(os.path.join(DATASET_DIR, "*", "sintetica_*.jpg")):
        os.remove(ruta)  # se regeneran desde cero

    fondos = fondos_vacios()
    print(f"Fondos vacíos: {len(fondos)}")
    for clase, carpeta in CARPETAS.items():
        recs = recortes(clase)
        reales = len(glob.glob(os.path.join(DATASET_DIR, carpeta, "pista_*.jpg")))
        n = max(0, objetivo - reales)
        print(f"{clase:<13} recortes: {len(recs):>2}   fotos de pista reales: {reales:>2}   sintéticas: {n}")
        for k in range(n):
            img = pegar(rng.choice(fondos), rng.choice(recs), rng)
            cv2.imwrite(os.path.join(DATASET_DIR, carpeta, f"sintetica_{k:02d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Listo: {objetivo} fotos de pista por clase ({DATASET_DIR}/<clase>/sintetica_*.jpg)")


if __name__ == "__main__":
    main()
