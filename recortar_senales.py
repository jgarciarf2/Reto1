import csv
import os
import random
import shutil
import sys

import cv2

from localizador_color import COLOR_DE_CLASE, candidatos, recorte_cuadrado

# ============================================================
# Arma el dataset de la CNN con RECORTES de la señal.
#
# En las fotos originales la señal ocupa una parte pequeña (el resto es el
# robot, la línea, la mano...), y la red aprende de todo eso. Aquí se recorta
# alrededor de la señal, que se localiza por color (localizador_color.py), y
# la CNN solo ve la señal. Escribe en 5_cnns/dataset_senal/<clase>/ :
#
#   NOMBRE.jpg        recorte cuadrado de la señal (256x256, sin deformar)
#   NOMBRE_c1.jpg     dos variantes con otro margen y desplazamiento, para
#   NOMBRE_c2.jpg     que aprenda más (solo se usan para entrenar)
#   NOMBRE_f1.jpg     la FOTO COMPLETA reducida a 256x256, para que la red también
#                     acierte cuando el localizador no encuentra la señal
#
# y la clase 4 (SIN_SENAL) con lo que NO es señal, que es lo que la CNN verá
# en vivo cuando el localizador proponga algo equivocado:
#   NOMBRE_n1.jpg     partes del robot que se parecen a una señal (pilas verdes
#                     y amarillas...), y trozos de pista, línea y fondo
#
# Las fotos donde no se encuentra la señal (tapada por la mano, cortada o que
# ni sale) no se usan, ni recortadas ni completas (puede que ni se vea la señal,
# y la etiqueta sería falsa), y se listan en _revisar.txt.
#
# Uso: python recortar_senales.py
# ============================================================

ORIGEN = "5_cnns/dataset"
DESTINO = "5_cnns/dataset_senal"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEMILLA = 42
LADO = 256
NEGATIVOS_MAX = 300          # imágenes de SIN_SENAL (sin contar variantes)
PARCHES_POR_FOTO_VACIA = 4   # trozos al azar de cada foto sin señal


def _interseccion(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ancho = min(ax + aw, bx + bw) - max(ax, bx)
    alto = min(ay + ah, by + bh) - max(ay, by)
    return max(0, ancho) * max(0, alto)


def _parche_al_azar(imagen, evitar, rng):
    """Cuadrado al azar que no toque la señal (`evitar`, si hay). None si no encuentra."""
    alto, ancho = imagen.shape[:2]
    for _ in range(25):
        lado = int(min(alto, ancho) * rng.uniform(0.10, 0.40))
        x = rng.randint(0, ancho - lado)
        y = rng.randint(0, alto - lado)
        if evitar is None or _interseccion((x, y, lado, lado), evitar) < 0.02 * lado * lado:
            return (x, y, lado, lado)
    return None


def _guardar(ruta, imagen):
    cv2.imwrite(ruta, imagen, [cv2.IMWRITE_JPEG_QUALITY, 95])


def procesar():
    rng = random.Random(SEMILLA)
    if os.path.isdir(DESTINO):
        shutil.rmtree(DESTINO)
    for clase in "01234":
        os.makedirs(os.path.join(DESTINO, clase), exist_ok=True)

    cajas_csv = []
    revisar = []
    negativos = []     # (prioridad, nombre, imagen): 0 = parecidas a señal, 1 = al azar
    positivos = {c: 0 for c in "0123"}

    for clase in "01234":
        carpeta = os.path.join(ORIGEN, clase)
        archivos = sorted(f for f in os.listdir(carpeta)
                          if os.path.splitext(f)[1].lower() in EXTENSIONES)
        for archivo in archivos:
            base = os.path.splitext(archivo)[0]
            imagen = cv2.imread(os.path.join(carpeta, archivo),
                                cv2.IMREAD_REDUCED_COLOR_2 | cv2.IMREAD_IGNORE_ORIENTATION)
            if imagen is None:
                continue

            completa = cv2.resize(imagen, (LADO, LADO), interpolation=cv2.INTER_AREA)

            if clase == "4":  # pista sin señal: foto completa y trozos al azar
                _guardar(os.path.join(DESTINO, "4", f"{base}_f1.jpg"), completa)
                for i in range(1, PARCHES_POR_FOTO_VACIA + 1):
                    caja = _parche_al_azar(imagen, None, rng)
                    if caja:
                        negativos.append((1, f"{base}_n{i}", recorte_cuadrado(imagen, caja, 1.0)))
                continue

            color = COLOR_DE_CLASE[clase]
            cs = candidatos(imagen, (color,)) or candidatos(imagen, (color,), relajado=True)
            if not cs:
                revisar.append(f"{clase}/{archivo}")
                continue

            caja = cs[0]["caja"]
            cajas_csv.append([clase, archivo, *caja, imagen.shape[1], imagen.shape[0]])
            carpeta_out = os.path.join(DESTINO, clase)
            _guardar(os.path.join(carpeta_out, f"{base}.jpg"), recorte_cuadrado(imagen, caja))
            _guardar(os.path.join(carpeta_out, f"{base}_f1.jpg"), completa)
            for i in (1, 2):  # variantes: otro margen y el recuadro movido un poco
                recorte = recorte_cuadrado(
                    imagen, caja, margen=rng.uniform(1.2, 1.8),
                    desplazamiento=(rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12)))
                _guardar(os.path.join(carpeta_out, f"{base}_c{i}.jpg"), recorte)
            positivos[clase] += 1

            # Negativos de esta foto: lo demás que el localizador toma por señal
            # (pilas del robot, etc.) y un trozo de fondo al azar
            otros = [c for c in candidatos(imagen)
                     if _interseccion(c["caja"], caja) < 0.1 * c["caja"][2] * c["caja"][3]]
            for i, c in enumerate(otros[:2], 1):
                negativos.append((0, f"{base}_n{i}", recorte_cuadrado(imagen, c["caja"])))
            if rng.random() < 0.5:
                parche = _parche_al_azar(imagen, caja, rng)
                if parche:
                    negativos.append((1, f"{base}_n9", recorte_cuadrado(imagen, parche, 1.0)))

        print(f"clase {clase}: listo")

    # SIN_SENAL: 60 % cosas que se parecen a una señal y 40 % fondo al azar
    rng.shuffle(negativos)
    parecidos = [n for n in negativos if n[0] == 0][:int(NEGATIVOS_MAX * 0.6)]
    al_azar = [n for n in negativos if n[0] == 1][:NEGATIVOS_MAX - len(parecidos)]
    for _, nombre, imagen in parecidos + al_azar:
        _guardar(os.path.join(DESTINO, "4", f"{nombre}.jpg"), imagen)

    with open(os.path.join(DESTINO, "_cajas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "archivo", "x", "y", "ancho", "alto", "ancho_img", "alto_img"])
        w.writerows(cajas_csv)
    with open(os.path.join(DESTINO, "_revisar.txt"), "w", encoding="utf-8") as f:
        f.write("Fotos sin señal localizada (no se usaron). Revísalas a mano:\n")
        f.write("\n".join(revisar) + "\n")

    print("\nSeñales localizadas por clase:", positivos)
    print(f"Sin señal localizada (en _revisar.txt): {len(revisar)}")
    for clase in "01234":
        n = len(os.listdir(os.path.join(DESTINO, clase)))
        print(f"  {clase}: {n} imágenes")


if __name__ == "__main__":
    procesar()
