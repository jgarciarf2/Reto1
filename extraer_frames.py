import os
import sys

import cv2

import config_robot
from main import AplicacionDetector
from modelos_cnn import CLASES

# ============================================================
# Arma el dataset desde VIDEOS grabados con el iPhone sobre el
# carro (la misma vista que tendrá el robot en producción).
#
# El video se reproduce con el MISMO giro que usa main.py, y con
# estas teclas dices qué hay en pantalla; mientras una clase esté
# activa se guarda un frame cada 0.3 s en <dataset>/<carpeta>/:
#
#   0 = PARE   1 = SIGA   2 = VELOCIDAD_60   3 = ZONA_ESCOLAR
#   4 = SIN_SENAL (pista vacía, mano, etc.)
#   n = no guardar nada      ESPACIO = pausa      q = salir
#
# Uso: python extraer_frames.py video.MOV [dataset] [--rotar=-90]
# ============================================================

INTERVALO_S = 0.3


def extraer(ruta_video, dataset="5_cnns/dataset", grados=None):
    grados = config_robot.ROTACION_CAMARA if grados is None else grados
    rotacion = AplicacionDetector.ROTACIONES[grados]
    for i in range(len(CLASES)):
        os.makedirs(os.path.join(dataset, str(i)), exist_ok=True)

    cap = cv2.VideoCapture(ruta_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    base = os.path.splitext(os.path.basename(ruta_video))[0]
    activa, pausa, ultimo_t, guardados = None, False, -1.0, {}
    frame = None

    while cap.isOpened():
        if not pausa:
            ret, frame = cap.read()
            if not ret:
                break
            if rotacion is not None:
                frame = cv2.rotate(frame, rotacion)
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if activa is not None and t - ultimo_t >= INTERVALO_S:
                ultimo_t = t
                ruta = os.path.join(dataset, str(activa), f"{base}_{int(t * 1000):06d}.jpg")
                cv2.imwrite(ruta, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                guardados[activa] = guardados.get(activa, 0) + 1

        vista = frame.copy()
        etiqueta = "no guarda" if activa is None else f"GUARDANDO {activa}={CLASES[activa]}"
        cv2.putText(vista, etiqueta, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(vista, etiqueta, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 0) if activa is not None else (0, 255, 255), 2, cv2.LINE_AA)
        alto = 800
        cv2.imshow("Extraer frames", cv2.resize(vista, (int(vista.shape[1] * alto / vista.shape[0]), alto)))

        tecla = cv2.waitKey(1 if pausa else int(1000 / fps)) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord(" "):
            pausa = not pausa
        elif tecla == ord("n"):
            activa = None
        elif ord("0") <= tecla < ord("0") + len(CLASES):
            activa = tecla - ord("0")

    cap.release()
    cv2.destroyAllWindows()
    print("Guardados en esta corrida:")
    for i, c in enumerate(CLASES):
        total = len(os.listdir(os.path.join(dataset, str(i))))
        print(f"  {i} {c}: +{guardados.get(i, 0)} (total {total})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit("Uso: python extraer_frames.py video.MOV [dataset] [--rotar=-90]")
    grados = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--rotar=")), None)
    extraer(args[0], args[1] if len(args) > 1 else "5_cnns/dataset", grados)
