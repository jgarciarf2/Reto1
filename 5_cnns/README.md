# Entrenamiento y Comparación de CNNs con Keras (Transfer Learning)

Este proyecto provee un pipeline automatizado para clasificar imágenes utilizando modelos convolucionales preentrenados en ImageNet (`VGG16`, `VGG19`, `MobileNet`, `ResNet50`, `DenseNet121`) mediante Keras y TensorFlow.

Incluye preparación de datos, early stopping, exportación de modelos, métricas completas (`classification_report`, `F1-macro`, `F1-ponderado`), matriz de confusión y curvas de pérdida organizadas jerárquicamente por experimento.

---

## Estructura del Proyecto

```text
5_cnns/
├── .venv/                     # Entorno virtual de Python
├── dataset/                   # Dataset unificado por categorías
│   ├── 0/                     # cero_0.jpg, cero_1.jpg, ...
│   ├── 1/                     # uno_0.jpg, ...
│   └── ...                    # (hasta clase 9)
├── resultados/                # Salidas generadas por experimentos
│   └── <nombre_experimento>/
│       ├── resultados.csv     # Resumen comparativo de todos los modelos
│       ├── comparativa_val_loss.png
│       └── <nombre_modelo>/
│           ├── <nombre_modelo>.keras
│           ├── perdida_<nombre_modelo>.png
│           ├── matriz_confusion_<nombre_modelo>.png
│           └── reporte_clasificacion_<nombre_modelo>.txt
├── cnn_trainer.py             # Clase CNNTrainer con la lógica de entrenamiento y evaluación
├── main.py                    # Script principal de configuración y ejecución
├── particionador.py           # Utilidad para unificar splits (train/test) y renombrar archivos
├── requirements.txt           # Dependencias del proyecto
└── .gitignore                 # Exclusiones de Git
```

---

## Modelos CNN Soportados

Las arquitecturas disponibles se importan desde `keras.applications` con pesos congelados de ImageNet y una cabeza de clasificación personalizada (Global Average Pooling + Dropout 0.3 + Softmax):

1. **VGG16**
2. **VGG19**
3. **MobileNet**
4. **ResNet50**
5. **DenseNet121**

Cada arquitectura aplica automáticamente su función de preprocesamiento específica (`preprocess_input`).

---

## Requisitos e Instalación

### 1. Requisitos previos
- Python 3.10 o superior (recomendado Python 3.10).

### 2. Crear y activar el entorno virtual

En Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Preparación del Dataset

El pipeline espera que dentro de `dataset/` existan subcarpetas por cada clase (`0`, `1`, `2`, ...).

Si tu dataset original viene dividido en carpetas `train/` y `test/`, puedes unificarlo y renombrar las imágenes automáticamente ejecutando:

```powershell
python particionador.py
```

Esto dejará los archivos organizados por categoría y nombrados como `cero_0.jpg`, `uno_0.jpg`, etc., evitando colisiones de nombres.

---

## Uso y Ejecución

Edita [main.py](main.py) para configurar los modelos que deseas entrenar y los hiperparámetros:

```python
from cnn_trainer import CNNTrainer

nombre_exp = "experimento_1"

hiperparametros = {
    "max_epochs": 50,          # Máximo de épocas permitidas
    "patience": 5,             # Paciencia del Early Stopping sobre val_loss
    "batch_size": 32,          # Tamaño del lote
    "learning_rate": 1e-3,     # Tasa de aprendizaje
    "img_size": (224, 224),    # Resolución de entrada para las CNNs
    "val_split": 0.2,          # 20% para validación
    "test_split": 0.1,         # 10% para test
    "seed": 42,                # Semilla de reproducibilidad
}

trainer = CNNTrainer(
    dataset_dir="dataset",
    output_dir="resultados",
    experimento=nombre_exp,
    **hiperparametros,
)

modelos_a_entrenar = [
    "VGG16",
    "MobileNet",
    "ResNet50",
]

trainer.entrenar_todos(modelos_a_entrenar)
```

Para iniciar el entrenamiento:

```powershell
python main.py
```

---

## Salidas y Métricas Generadas

Para cada modelo entrenado dentro de `resultados/<experimento>/<modelo>/` se generan:

1. **`<modelo>.keras`**: Pesos del modelo restaurados en la mejor época (según menor `val_loss`).
2. **`perdida_<modelo>.png`**: Curvas de pérdida en entrenamiento vs validación, indicando la mejor época.
3. **`matriz_confusion_<modelo>.png`**: Matriz de confusión calculada sobre el conjunto de test.
4. **`reporte_clasificacion_<modelo>.txt`**: Classification Report de scikit-learn con `precision`, `recall`, `f1-score` y `support` por clase.
5. **`resultados.csv`**: Tabla consolidada en la raíz del experimento con las siguientes columnas:
   - `modelo`
   - `experimento`
   - `max_epochs`
   - `epoca_parada`
   - `mejor_epoca`
   - `train_loss`
   - `val_loss`
   - `accuracy`
   - `f1_ponderado`
   - `f1_macro`
   - `precision_macro`
   - `recall_macro`
   - `grafica_perdida`
   - `grafica_confusion`
   - `archivo_reporte`
6. **`comparativa_val_loss.png`**: Gráfica comparativa de curvas de validación cuando se entrena más de un modelo en la misma corrida.
