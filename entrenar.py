import os
import sys

from entrenador_cnn import CNNTrainer
from modelos_cnn import CLASES, IMG_SIZE

# Uso: python entrenar.py [experimento] [Modelo1 Modelo2 ...]
# Los resultados quedan en resultados/<experimento>/<Modelo>/.
# Por defecto solo los 2 mejores del experimento anterior (reto2_exp1).
nombre_exp = sys.argv[1] if len(sys.argv) > 1 else "reto2_exp3"
modelos = sys.argv[2:] or ["MobileNet", "ResNet50"]

DATASET_DIR = "5_cnns/dataset"  # 0=PARE, 1=SIGA, 2=VELOCIDAD_60, 3=ZONA_ESCOLAR (4=SIN_SENAL opcional)
carpetas = sorted(d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)))

trainer = CNNTrainer(
    dataset_dir=DATASET_DIR,
    output_dir="resultados",
    experimento=nombre_exp,
    nombres_clases=CLASES[:len(carpetas)],
    max_epochs=40,
    patience=10,       # early stopping: para si val_loss no mejora en 10 épocas (y se queda con la mejor)
    batch_size=16,
    learning_rate=1e-3,
    img_size=IMG_SIZE,
    val_split=0.2,
    test_split=0.15,
    seed=42,
)

# Entrena y compara; al final deja los 2 mejores (por validación) en seleccion.json
trainer.entrenar_todos(modelos)
