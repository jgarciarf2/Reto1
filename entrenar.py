import sys

from entrenador_cnn import CNNTrainer
from modelos_cnn import CLASES, DATASET_DIR, IMG_SIZE, MODELOS

# Uso: python entrenar.py [experimento] [Modelo1 Modelo2 ...]
nombre_exp = sys.argv[1] if len(sys.argv) > 1 else "reto2_exp1"
modelos = sys.argv[2:] or list(MODELOS)  # por defecto las 5 CNN de clase

trainer = CNNTrainer(
    dataset_dir=DATASET_DIR,     # carpetas 0, 1, 2, 3 con las fotos de cada señal
    output_dir="resultados",
    experimento=nombre_exp,
    nombres_clases=CLASES,
    max_epochs=40,
    patience=6,
    batch_size=16,
    learning_rate=1e-3,
    img_size=IMG_SIZE,
    val_split=0.2,
    test_split=0.15,
    seed=42,
)

# Entrena y compara todos; al final deja los 2 mejores (por validación) en seleccion.json
trainer.entrenar_todos(modelos)
