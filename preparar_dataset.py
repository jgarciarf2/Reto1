import os
import random
import sys

import cv2

from modelos_cnn import IMG_SIZE

# ============================================================
# Prepara el dataset para entrenar la CNN a 256x256.
#
# Lee las fotos originales (5_cnns/dataset/<clase>/) y escribe en
# 5_cnns/dataset_256/<clase>/ :
#
#   NOMBRE.jpg      la foto completa reducida a 256x256 (se aplasta igual
#                   que hace el predictor con cada frame de la cámara)
#   NOMBRE_c1.jpg   recortes al azar de la misma foto (85-95 % del ancho
#   NOMBRE_c2.jpg   y del alto), también a 256x256
#
# Las fotos originales pesan varios MB: reducirlas una sola vez hace el
# entrenamiento mucho más rápido. Los recortes dan variedad de encuadre
# (la señal queda en otro lugar y a otro tamaño) para que la red aprenda más.
# El entrenador solo usa los recortes para entrenar, nunca para validar ni
# probar, y mantiene juntos la foto y sus recortes (ver entrenador_cnn.py).
#
# Uso: python preparar_dataset.py [recortes_por_foto]
# ============================================================

ORIGEN = "5_cnns/dataset"
DESTINO = "5_cnns/dataset_256"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRACCION_MIN, FRACCION_MAX = 0.85, 0.95
SEMILLA = 42


def a_256(imagen):
    # INTER_AREA es la mejor opción al reducir fotos muy grandes
    return cv2.resize(imagen, IMG_SIZE[::-1], interpolation=cv2.INTER_AREA)


def recorte_al_azar(imagen, rng):
    alto, ancho = imagen.shape[:2]
    h = int(alto * rng.uniform(FRACCION_MIN, FRACCION_MAX))
    w = int(ancho * rng.uniform(FRACCION_MIN, FRACCION_MAX))
    y = rng.randint(0, alto - h)
    x = rng.randint(0, ancho - w)
    return imagen[y:y + h, x:x + w]


def preparar(recortes_por_foto=2):
    rng = random.Random(SEMILLA)
    clases = sorted(d for d in os.listdir(ORIGEN) if os.path.isdir(os.path.join(ORIGEN, d)))
    total_fotos = total_recortes = 0

    for clase in clases:
        carpeta_out = os.path.join(DESTINO, clase)
        os.makedirs(carpeta_out, exist_ok=True)
        archivos = sorted(f for f in os.listdir(os.path.join(ORIGEN, clase))
                          if os.path.splitext(f)[1].lower() in EXTENSIONES)

        for archivo in archivos:
            # Sin rotar por EXIF, igual que en el entrenamiento y en el predictor
            imagen = cv2.imread(os.path.join(ORIGEN, clase, archivo),
                                cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            if imagen is None:
                print(f"  (no se pudo leer {clase}/{archivo}, se omite)")
                continue

            nombre = os.path.splitext(archivo)[0]
            cv2.imwrite(os.path.join(carpeta_out, f"{nombre}.jpg"), a_256(imagen),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            total_fotos += 1

            for i in range(1, recortes_por_foto + 1):
                recorte = a_256(recorte_al_azar(imagen, rng))
                cv2.imwrite(os.path.join(carpeta_out, f"{nombre}_c{i}.jpg"), recorte,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                total_recortes += 1

        print(f"clase {clase}: {len(archivos)} fotos")

    print(f"\nListo en {DESTINO}: {total_fotos} fotos + {total_recortes} recortes "
          f"= {total_fotos + total_recortes} imágenes de {IMG_SIZE[0]}x{IMG_SIZE[1]}")


if __name__ == "__main__":
    preparar(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
