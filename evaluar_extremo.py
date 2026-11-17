import csv
import os
import sys
from collections import Counter

import cv2

import config_robot
from modelos_cnn import CLASES
from predictor_senales import PredictorSenales

# ============================================================
# Evalúa el sistema COMPLETO (localizar -> recortar -> clasificar, igual que
# en el robot) sobre las FOTOS ORIGINALES que cayeron en la partición de
# prueba durante el entrenamiento (particiones.csv), o sea, fotos que la red
# no vio. Compara tres formas de mirar la imagen:
#
#   recortes + respaldo   localiza la señal y, si no encuentra nada, mira el frame completo
#   solo recortes         si no encuentra nada, dice que no hay señal
#   completo              siempre el frame completo
#
# Uso: python evaluar_extremo.py [carpeta_resultados]   (por defecto resultados_mixto)
# ============================================================

DATASET = "5_cnns/dataset"
NOMBRES = ["PARE", "SIGA", "VELOCIDAD_60", "ZONA_ESCOLAR", "SIN_SENAL"]


def fotos_de_prueba(carpeta):
    """[(ruta_original, clase_real)] de las fotos completas (_f1) de la partición de prueba."""
    pruebas = []
    with open(os.path.join(carpeta, "particiones.csv"), encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            nombre = os.path.splitext(os.path.basename(fila["archivo"].replace("\\", "/")))[0]
            if fila["particion"] != "test" or not nombre.endswith("_f1"):
                continue
            base = nombre[:-3]
            clase = str(NOMBRES.index(fila["clase"]) if fila["clase"] in NOMBRES else int(fila["clase"]))
            for ext in (".jpg", ".jpeg", ".png"):
                ruta = os.path.join(DATASET, clase, base + ext)
                if os.path.exists(ruta):
                    pruebas.append((ruta, int(clase)))
                    break
    return pruebas


def evaluar(predictor, pruebas, modo, respaldo):
    predictor.modo = modo
    config_robot.CNN_RESPALDO_COMPLETO = respaldo
    aciertos, confusion, por_clase = 0, Counter(), Counter()
    for ruta, real in pruebas:
        # Se lee a la mitad, como se prepararon las fotos de entrenamiento
        foto = cv2.imread(ruta, cv2.IMREAD_REDUCED_COLOR_2 | cv2.IMREAD_IGNORE_ORIENTATION)
        probs = predictor.probabilidades(foto)
        pred = predictor.clases.index(max(probs, key=probs.get))
        aciertos += pred == real
        confusion[(real, pred)] += 1
        por_clase[real] += 1
    return aciertos, confusion, por_clase


if __name__ == "__main__":
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "resultados_mixto"
    predictor = PredictorSenales(carpeta=carpeta)
    pruebas = fotos_de_prueba(carpeta)
    print(f"{len(pruebas)} fotos de prueba (no vistas), modelo {predictor.nombre} de {carpeta}\n")

    for etiqueta, modo, respaldo in [("recortes + respaldo", "recortes", True),
                                      ("solo recortes", "recortes", False),
                                      ("completo", "completo", False)]:
        aciertos, confusion, por_clase = evaluar(predictor, pruebas, modo, respaldo)
        print(f"== {etiqueta}: {aciertos}/{len(pruebas)} = {aciertos / len(pruebas):.3f}")
        for real in sorted(por_clase):
            bien = confusion[(real, real)]
            errores = ", ".join(f"{NOMBRES[p]}×{n}" for (r, p), n in sorted(confusion.items())
                                if r == real and p != real)
            print(f"   {NOMBRES[real]:13s} {bien}/{por_clase[real]}" + (f"   se confundió con: {errores}" if errores else ""))
        print()
