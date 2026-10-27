import sys

from entrenador_cnn import CNNTrainer
from modelos_cnn import IMG_SIZE

# Uso: python entrenar.py [experimento] [Modelo1 Modelo2 ...]
nombre_exp = sys.argv[1] if len(sys.argv) > 1 else "reto2_exp1"
modelos = sys.argv[2:] or ["MobileNet", "VGG16", "ResNet50", "DenseNet121"]

trainer = CNNTrainer(
    dataset_dir="dataset",       # carpetas 0, 1, 2, 3 con las fotos de cada señal
    output_dir="resultados",
    experimento=nombre_exp,
    max_epochs=30,
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
