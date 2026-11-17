import sys

from entrenador_cnn import CNNTrainer
from modelos_cnn import IMG_SIZE

# Uso: python entrenar.py [Modelo1 Modelo2 ...] [--dataset=carpeta] [--salida=carpeta]
# Los modelos quedan en <salida>/<Modelo>/ (se sobrescriben en cada corrida).
#   python entrenar.py MobileNet                       -> fotos completas (5_cnns/dataset_256)
#   python entrenar.py MobileNet --dataset=5_cnns/dataset_senal --salida=resultados_senal
#                                                      -> recortes de la señal (recortar_senales.py)
def opcion(nombre, defecto):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{nombre}=")), defecto)


modelos = [a for a in sys.argv[1:] if not a.startswith("--")] or ["MobileNet"]

trainer = CNNTrainer(
    dataset_dir=opcion("dataset", "5_cnns/dataset_256"),  # 0=PARE, 1=SIGA, 2=VELOCIDAD_60, 3=ZONA_ESCOLAR, 4=SIN_SENAL
    output_dir=opcion("salida", "resultados"),
    experimento="reto2",
    max_epochs=40,
    patience=5,
    batch_size=32,
    learning_rate=1e-3,
    img_size=IMG_SIZE,
    val_split=0.2,
    test_split=0.1,
    seed=42,
)

# Entrena y compara todos; al final deja los 2 mejores en seleccion.json
trainer.entrenar_todos(modelos)
