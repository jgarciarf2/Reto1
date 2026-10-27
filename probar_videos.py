"""
Pasa la CNN por los frames de un video de la pista y muestra, segundo a
segundo, qué predice y qué señales llegaría a obedecer ControlCNN. Sirve
para ver si la CNN "inventa" señales cuando la cámara solo ve la pista.

    python probar_videos.py                          # video de prueba del dataset
    python probar_videos.py otro.mp4 --modelo ResNet50 --rotar
"""
import argparse
from collections import Counter

import cv2

import config_robot
from control_cnn import ControlCNN
from main import AplicacionDetector
from predictor_senales import PredictorSenales

VIDEO_PRUEBA = "5_cnns/dataset/WhatsApp Video 2026-10-27 at 11.26.30 AM.mp4"

ABREVIATURAS = {"PARE": "PARE", "SIGA": "SIGA", "VELOCIDAD_60": "60", "ZONA_ESCOLAR": "ZONA"}


def _texto(img, texto, y, color, escala=0.7):
    cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2, cv2.LINE_AA)


def _anotar(frame, t, clase, prob, seguro, control, aviso):
    """Dibuja sobre el frame la predicción y el estado del robot."""
    _texto(frame, f"{t:5.1f}s  CNN: {clase} {prob * 100:.0f}%{' *' if seguro else ''}", 28,
           (0, 255, 255) if seguro else (200, 200, 200))
    estado = "DETENIDO" if control.detenido else "EN MARCHA"
    _texto(frame, f"{estado}  vel {control.velocidad}", 58,
           (0, 0, 255) if control.detenido else (0, 220, 0))
    if aviso:
        _texto(frame, f"OBEDECE: {aviso}", frame.shape[0] - 15, (255, 0, 255), 0.9)


def probar(predictor, ruta, rotar=False, intervalo_s=config_robot.INTERVALO_CNN_S, salida=None):
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cada_n = max(1, round(fps * intervalo_s))  # mismo ritmo de consulta que en el robot
    control = ControlCNN()
    linea_tiempo, eventos, sobre_umbral = [], [], Counter()
    escritor, ultimo, aviso, aviso_hasta = None, None, None, 0
    k = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        k += 1
        t = k / fps
        if rotar:
            frame = cv2.rotate(frame, AplicacionDetector.ROTACION_CAMARA)
        if k % cada_n == 0:
            probs = predictor.probabilidades(frame)
            clase = max(probs, key=probs.get)
            prob = probs[clase]
            seguro = prob >= config_robot.CONFIANZA_MINIMA[clase]
            sobre_umbral[clase] += seguro
            linea_tiempo.append((t, clase, prob, seguro))
            ultimo = (clase, prob, seguro)
            senal = control.actualizar(probs)
            if senal:
                estado = "DETENIDO" if control.detenido else "EN MARCHA"
                eventos.append(f"{t:5.1f}s  {senal:<13} -> {estado}, velocidad {control.velocidad}")
                aviso, aviso_hasta = senal, t + 1.5
        if salida:
            if escritor is None:
                alto, ancho = frame.shape[:2]
                escritor = cv2.VideoWriter(salida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
            _anotar(frame, t, *ultimo, control, aviso if t < aviso_hasta else None)
            escritor.write(frame)
    cap.release()
    if escritor:
        escritor.release()
    return linea_tiempo, eventos, sobre_umbral


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=VIDEO_PRUEBA)
    parser.add_argument("--experimento", default=config_robot.EXPERIMENTO_CNN)
    parser.add_argument("--modelo", default=None)
    parser.add_argument("--rotar", action="store_true",
                        help="girar el frame como en main.py (stream del celular en vertical)")
    parser.add_argument("--guardar", metavar="SALIDA.mp4",
                        help="guarda el video con la predicción y el estado dibujados")
    args = parser.parse_args()

    predictor = PredictorSenales(args.experimento, modelo=args.modelo)
    linea_tiempo, eventos, sobre_umbral = probar(predictor, args.video, args.rotar,
                                                 salida=args.guardar)
    if args.guardar:
        print(f"Video anotado guardado en: {args.guardar}")

    print(f"Modelo: {predictor.nombre}   Video: {args.video}")
    print("\nLínea de tiempo (cada ~0.5 s; * = supera el umbral de confianza):")
    paso = max(1, round(0.5 / config_robot.INTERVALO_CNN_S))
    celdas = [f"{t:5.1f}s {ABREVIATURAS[c]:>4} {p * 100:3.0f}%{'*' if s else ' '}"
              for t, c, p, s in linea_tiempo[::paso]]
    for i in range(0, len(celdas), 6):
        print("   ".join(celdas[i:i + 6]))

    print("\nPredicciones sobre el umbral:", dict(sobre_umbral))
    print("\nSeñales que el control obedecería:")
    print("\n".join(eventos) or "  ninguna")


if __name__ == "__main__":
    main()
