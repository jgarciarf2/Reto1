import csv
import json
import os
import re
import time

import keras
from keras import layers, models
import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica para guardar figuras en disco
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
from PIL import Image

from modelos_cnn import MODELOS


def _dhash(ruta):
    """Huella de 64 bits de la foto: compara el brillo de píxeles vecinos en una miniatura 9x8."""
    img = Image.open(ruta)
    img.draft("L", (160, 160))  # decodifica el JPEG ya reducido (mucho más rápido)
    img = img.convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _segundo_de_captura(nombre):
    """Segundo del día en que se tomó la foto, leído del nombre del archivo (o None)."""
    m = re.search(r"\d{8}_(\d{2})(\d{2})(\d{2})", nombre)  # 20261027_095759.jpg
    if m:
        h, mi, s = map(int, m.groups())
        return h * 3600 + mi * 60 + s
    m = re.search(r"at (\d+)\.(\d+)\.(\d+) (AM|PM)", nombre)  # WhatsApp Image ... at 9.43.44 AM
    if m:
        h, mi, s = map(int, m.groups()[:3])
        h = h % 12 + (12 if m.group(4) == "PM" else 0)
        return h * 3600 + mi * 60 + s
    return None


def agrupar_casi_identicas(file_paths, labels, umbral_hash=10):
    """
    Une en un mismo grupo las fotos de la misma clase que son casi iguales:
    las que difieren en <= `umbral_hash` bits de su dHash, o las que se
    tomaron en el mismo segundo (ráfaga). Devuelve el id de grupo de cada foto.
    """
    padre = list(range(len(file_paths)))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    hashes = np.array([_dhash(p) for p in file_paths])
    segundos = [_segundo_de_captura(os.path.basename(p)) for p in file_paths]
    for i in range(len(file_paths)):
        for j in range(i + 1, len(file_paths)):
            if labels[i] != labels[j]:
                continue
            parecidas = np.count_nonzero(hashes[i] != hashes[j]) <= umbral_hash
            rafaga = segundos[i] is not None and segundos[i] == segundos[j]
            if parecidas or rafaga:
                padre[raiz(i)] = raiz(j)

    raices = [raiz(i) for i in range(len(file_paths))]
    ids = {r: k for k, r in enumerate(dict.fromkeys(raices))}
    return np.array([ids[r] for r in raices])


class CNNTrainer:
    """Entrena CNN base de Keras (transfer learning) sobre un dataset en carpetas
    `dataset_dir/0`, `dataset_dir/1`, ... (una carpeta por clase)."""

    def __init__(self, dataset_dir, img_size=(224, 224), batch_size=32,
                 val_split=0.2, test_split=0.1, patience=5, max_epochs=100,
                 learning_rate=1e-3, seed=42, output_dir="resultados",
                 experimento="experimento_1", nombres_clases=None, umbral_hash=10):
        self.dataset_dir = dataset_dir
        self.nombres_clases = nombres_clases
        self.umbral_hash = umbral_hash
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.patience = patience
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.output_dir = output_dir
        self.experimento = experimento
        self.exp_dir = os.path.join(self.output_dir, self.experimento)
        self.historiales = {}
        self.reportes = {}
        os.makedirs(self.exp_dir, exist_ok=True)
        self._cargar_datos()

    def _cargar_datos(self):
        if self.val_split <= 0 or self.test_split <= 0:
            raise ValueError("val_split y test_split deben ser mayores que 0.")
        if self.val_split + self.test_split >= 1.0:
            raise ValueError("La suma de val_split + test_split debe ser menor que 1.0.")

        class_names = sorted(
            [d for d in os.listdir(self.dataset_dir)
             if os.path.isdir(os.path.join(self.dataset_dir, d))]
        )
        if not class_names:
            raise ValueError(f"No se encontraron carpetas de clase en '{self.dataset_dir}'.")

        extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        file_paths = []
        labels = []

        for idx, class_name in enumerate(class_names):
            class_dir = os.path.join(self.dataset_dir, class_name)
            for root, _, files in os.walk(class_dir):
                for file_name in files:
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext in extensiones:
                        file_paths.append(os.path.join(root, file_name))
                        labels.append(idx)

        if not file_paths:
            raise ValueError(f"No se encontraron imágenes válidas en '{self.dataset_dir}'.")

        self.carpetas = class_names
        # Nombres legibles (PARE, SIGA, ...) para reportes y matrices
        self.class_names = self.nombres_clases or class_names
        self.num_classes = len(self.class_names)

        # Las fotos se tomaron en ráfagas: varias son casi iguales. Si una
        # queda en train y su gemela en test, el test deja de medir algo
        # real. Por eso se parte por grupos de fotos casi idénticas y cada
        # grupo cae completo en una sola partición.
        grupos = agrupar_casi_identicas(file_paths, labels, self.umbral_hash)
        idx = np.arange(len(file_paths))
        labels_np = np.array(labels)

        n_test = max(2, round(1 / self.test_split))
        sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=self.seed)
        resto, test_idx = next(sgkf.split(idx, labels_np, grupos))

        val_rel = self.val_split / (1.0 - self.test_split)
        n_val = max(2, round(1 / val_rel))
        sgkf = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=self.seed)
        tr, va = next(sgkf.split(resto, labels_np[resto], grupos[resto]))
        train_idx, val_idx = resto[tr], resto[va]

        self._guardar_particiones(file_paths, labels, grupos,
                                  {"train": train_idx, "val": val_idx, "test": test_idx})

        def tomar(ids):
            return [file_paths[i] for i in ids], [labels[i] for i in ids]

        self.train_raw = self._crear_dataset(*tomar(train_idx), entrenar=True)
        self.val_raw = self._crear_dataset(*tomar(val_idx))
        self.test_raw = self._crear_dataset(*tomar(test_idx))

    def _guardar_particiones(self, file_paths, labels, grupos, particiones):
        """Deja en disco qué foto quedó en cada partición (las mismas para todos los modelos)."""
        ruta = os.path.join(self.exp_dir, "particiones.csv")
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["particion", "clase", "grupo", "archivo"])
            for nombre, ids in particiones.items():
                for i in ids:
                    w.writerow([nombre, self.class_names[labels[i]], grupos[i], file_paths[i]])
        print(f"Grupos de fotos casi idénticas: {len(set(grupos))} (de {len(file_paths)} fotos)")
        for nombre, ids in particiones.items():
            conteo = np.bincount([labels[i] for i in ids], minlength=self.num_classes)
            detalle = ", ".join(f"{c}={n}" for c, n in zip(self.class_names, conteo))
            print(f"  {nombre:<5} {len(ids):>3} fotos  ({detalle})")
        print(f"Particiones guardadas en: {ruta}")

    def _crear_dataset(self, paths, labels, entrenar=False):
        path_ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

        def _cargar(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, self.img_size)
            img = tf.cast(img, tf.float32)
            return img, label

        # Las fotos originales pesan mucho: se decodifican una sola vez y se
        # guardan ya reducidas a img_size.
        ds = path_ds.map(_cargar, num_parallel_calls=tf.data.AUTOTUNE).cache()
        if entrenar:
            self._n_train = len(paths)
        return ds

    def _preparar(self, ds, preprocess, entrenar=False):
        if entrenar:
            ds = ds.shuffle(self._n_train, seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.batch(self.batch_size)
        if entrenar:
            aumentos = self._aumentos()
            ds = ds.map(lambda x, y: (aumentos(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x, y: (preprocess(tf.cast(x, tf.float32)), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    def _aumentos(self):
        """
        Variaciones aleatorias sobre los píxeles originales (0-255), solo al
        entrenar. En las fotos la señal aparece girada en cualquier ángulo
        (incluso al revés), así que se rota en todo el círculo. Brillo y
        contraste simulan la luz cambiante de la pista. No hay flip: invertiría
        el texto de PARE / SIGA y el "60".
        """
        return keras.Sequential([
            layers.RandomRotation(0.5, fill_mode="reflect", seed=self.seed),
            layers.RandomZoom((-0.2, 0.25), fill_mode="reflect", seed=self.seed),
            layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=self.seed),
            layers.RandomBrightness(0.2, value_range=(0, 255), seed=self.seed),
            layers.RandomContrast(0.25, seed=self.seed),
        ], name="aumentos")

    def _construir(self, nombre):
        constructor, _ = MODELOS[nombre]
        base = constructor(weights="imagenet", include_top=False,
                           input_shape=self.img_size + (3,))
        base.trainable = False
        entrada = layers.Input(self.img_size + (3,))
        x = base(entrada, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        salida = layers.Dense(self.num_classes, activation="softmax")(x)
        modelo = models.Model(entrada, salida, name=nombre)
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return modelo

    def entrenar(self, nombre):
        """Entrena un modelo con early stopping sobre val_loss y devuelve un dict de resultados."""
        if nombre not in MODELOS:
            raise ValueError(f"Modelo '{nombre}' no soportado: {list(MODELOS)}")
        prep = MODELOS[nombre][1]
        train = self._preparar(self.train_raw, prep, entrenar=True)
        val = self._preparar(self.val_raw, prep)
        test = self._preparar(self.test_raw, prep)

        modelo = self._construir(nombre)
        # Para si val_loss no mejora (se estanca o sube) durante `patience` épocas
        early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience,
            restore_best_weights=True, verbose=1)
        hist = modelo.fit(train, validation_data=val,
                          epochs=self.max_epochs, callbacks=[early], verbose=2)

        y_true = np.concatenate([y.numpy() for _, y in test])
        y_pred = np.argmax(modelo.predict(test, verbose=0), axis=1)
        epocas = len(hist.history["loss"])
        mejor = int(np.argmin(hist.history["val_loss"])) + 1

        # Carpeta específica para este modelo: resultados/experimento/nombremodelo
        model_dir = os.path.join(self.exp_dir, nombre)
        os.makedirs(model_dir, exist_ok=True)

        # Generar y guardar gráfica de pérdida y matriz de confusión
        grafica_perdida = self._graficar_perdida(nombre, hist.history, mejor, model_dir)
        grafica_cm = self._graficar_matriz_confusion(nombre, y_true, y_pred, model_dir)
        self.historiales[nombre] = hist.history

        # Generar y guardar classification report de sklearn
        reporte_str, reporte_dict = self._guardar_classification_report(nombre, y_true, y_pred, model_dir, epocas, mejor)

        # Guardar el modelo entrenado dentro de su propia carpeta
        modelo_path = os.path.join(model_dir, f"{nombre}.keras")
        modelo.save(modelo_path)
        print(f"Modelo guardado: {modelo_path}")

        ms_por_foto = self._medir_tiempo(modelo)
        print(f"Tiempo de respuesta: {ms_por_foto:.1f} ms por foto")

        etiquetas_indices = list(range(self.num_classes))
        self.reportes[nombre] = reporte_dict
        return {
            "modelo": nombre,
            "experimento": self.experimento,
            "max_epochs": self.max_epochs,
            "epoca_parada": epocas,
            "mejor_epoca": mejor,
            "train_loss": hist.history["loss"][mejor - 1],
            "val_loss": hist.history["val_loss"][mejor - 1],
            "accuracy": reporte_dict.get("accuracy", 0.0),
            "f1_ponderado": f1_score(y_true, y_pred, labels=etiquetas_indices, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, labels=etiquetas_indices, average="macro", zero_division=0),
            "precision_macro": reporte_dict.get("macro avg", {}).get("precision", 0.0),
            "recall_macro": reporte_dict.get("macro avg", {}).get("recall", 0.0),
            "ms_por_foto": ms_por_foto,
            "grafica_perdida": grafica_perdida,
            "grafica_confusion": grafica_cm,
            "archivo_reporte": f"reporte_clasificacion_{nombre}.txt",
        }

    def _medir_tiempo(self, modelo, repeticiones=30):
        """Milisegundos que tarda el modelo en clasificar UNA foto (como en el robot)."""
        x = tf.zeros((1,) + self.img_size + (3,))
        for _ in range(5):  # calentamiento
            modelo(x, training=False)
        inicio = time.perf_counter()
        for _ in range(repeticiones):
            modelo(x, training=False)
        return (time.perf_counter() - inicio) / repeticiones * 1000

    def _guardar_classification_report(self, nombre, y_true, y_pred, model_dir, epocas, mejor_epoca):
        """Calcula el classification_report de scikit-learn, lo muestra en consola y lo guarda en disco."""
        etiquetas_indices = list(range(self.num_classes))
        etiquetas_nombres = [str(c) for c in self.class_names]
        reporte_str = classification_report(
            y_true, y_pred,
            labels=etiquetas_indices,
            target_names=etiquetas_nombres,
            digits=4,
            zero_division=0,
        )
        reporte_dict = classification_report(
            y_true, y_pred,
            labels=etiquetas_indices,
            target_names=etiquetas_nombres,
            output_dict=True,
            zero_division=0,
        )

        print(f"\n--- Reporte de Clasificación ({nombre}) ---\n{reporte_str}")

        ruta_txt = os.path.join(model_dir, f"reporte_clasificacion_{nombre}.txt")
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(f"Experimento: {self.experimento}\n")
            f.write(f"Modelo: {nombre}\n")
            f.write(f"Épocas totales entrenadas: {epocas}\n")
            f.write(f"Mejor época seleccionada: {mejor_epoca}\n")
            f.write("=" * 60 + "\n\n")
            f.write(reporte_str)

        print(f"Reporte de clasificación guardado: {ruta_txt}")
        return reporte_str, reporte_dict

    def _graficar_perdida(self, nombre, history, mejor_epoca, model_dir):
        """Genera y guarda una gráfica de pérdida (train vs val) para el modelo."""
        train_loss = history["loss"]
        val_loss = history["val_loss"]
        epocas = list(range(1, len(train_loss) + 1))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epocas, train_loss, label="Pérdida Entrenamiento (Train Loss)", color="#1f77b4", linewidth=2)
        ax.plot(epocas, val_loss, label="Pérdida Validación (Val Loss)", color="#ff7f0e", linewidth=2)

        # Resaltar la mejor época seleccionada por EarlyStopping
        ax.axvline(x=mejor_epoca, color="red", linestyle="--", alpha=0.7,
                   label=f"Mejor época ({mejor_epoca})")
        ax.scatter([mejor_epoca], [val_loss[mejor_epoca - 1]], color="red", s=50, zorder=5)

        ax.set_title(f"Curva de Pérdida - {nombre}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Épocas", fontsize=11)
        ax.set_ylabel("Pérdida (Cross-Entropy)", fontsize=11)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        nombre_archivo = f"perdida_{nombre}.png"
        ruta_grafica = os.path.join(model_dir, nombre_archivo)
        plt.savefig(ruta_grafica, dpi=300)
        plt.close(fig)
        print(f"Gráfica de pérdida guardada: {ruta_grafica}")
        return nombre_archivo

    def _graficar_matriz_confusion(self, nombre, y_true, y_pred, model_dir):
        """Genera y guarda la matriz de confusión de las predicciones sobre el conjunto de test."""
        fig, ax = plt.subplots(figsize=(8, 7))
        etiquetas_indices = list(range(self.num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=etiquetas_indices)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")

        ax.set_title(f"Matriz de Confusión - {nombre}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicción", fontsize=11)
        ax.set_ylabel("Etiqueta Real", fontsize=11)
        plt.tight_layout()

        nombre_archivo = f"matriz_confusion_{nombre}.png"
        ruta_cm = os.path.join(model_dir, nombre_archivo)
        plt.savefig(ruta_cm, dpi=300)
        plt.close(fig)
        print(f"Matriz de confusión guardada: {ruta_cm}")
        return nombre_archivo

    def _graficar_comparativa(self, historiales):
        """Genera una gráfica comparativa con las pérdidas de validación de todos los modelos."""
        fig, ax = plt.subplots(figsize=(9, 5))
        for nombre, hist in historiales.items():
            val_loss = hist["val_loss"]
            epocas = list(range(1, len(val_loss) + 1))
            ax.plot(epocas, val_loss, label=f"{nombre}", linewidth=2)

        ax.set_title(f"Comparativa de Validación - Experimento: {self.experimento}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Épocas", fontsize=11)
        ax.set_ylabel("Pérdida de Validación (Val Loss)", fontsize=11)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        ruta = os.path.join(self.exp_dir, "comparativa_val_loss.png")
        plt.savefig(ruta, dpi=300)
        plt.close(fig)
        print(f"Gráfica comparativa guardada: {ruta}")

    def entrenar_todos(self, nombres=None, csv_name="resultados.csv"):
        """Entrena los modelos indicados y guarda el CSV resumen en la carpeta del experimento."""
        resultados = []
        ruta = os.path.join(self.exp_dir, csv_name)
        modelos_seleccionados = nombres or list(MODELOS)

        for nombre in modelos_seleccionados:
            print(f"\n===== Entrenando {nombre} (Experimento: {self.experimento}) =====")
            res = self.entrenar(nombre)
            resultados.append(res)
            keras.backend.clear_session()
            self._guardar_csv(resultados, ruta)  # se guarda tras cada modelo

        if len(self.historiales) > 1:
            self._graficar_comparativa(self.historiales)

        self._guardar_seleccion(resultados)

        print(f"\nResumen del experimento guardado en: {ruta}")
        return resultados

    def _guardar_seleccion(self, resultados, top=2):
        """Elige los `top` mejores modelos por val_loss y guarda la selección + las clases."""
        ranking = sorted(resultados, key=lambda r: r["val_loss"])[:top]
        seleccion = {
            "experimento": self.experimento,
            "clases": self.class_names,
            "img_size": list(self.img_size),
            "mejores": [{"modelo": r["modelo"], "val_loss": r["val_loss"],
                         "accuracy": r["accuracy"], "f1_macro": r["f1_macro"]}
                        for r in ranking],
        }
        ruta = os.path.join(self.exp_dir, "seleccion.json")
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(seleccion, f, indent=2, ensure_ascii=False)
        print("\nMejores modelos (menor val_loss):")
        for i, r in enumerate(ranking, 1):
            print(f"  {i}. {r['modelo']}  val_loss={r['val_loss']:.4f}  f1_macro={r['f1_macro']:.4f}")
        print(f"Selección guardada en: {ruta}")
        self._guardar_tabla_mejores(ranking)

    def _guardar_tabla_mejores(self, ranking):
        """Tabla comparativa (sobre test) de los mejores: por clase y promedio macro."""
        filas = []
        for r in ranking:
            rep = self.reportes[r["modelo"]]
            for clase in list(self.class_names) + ["macro avg"]:
                m = rep[clase]
                filas.append({
                    "modelo": r["modelo"],
                    "clase": "PROMEDIO MACRO" if clase == "macro avg" else clase,
                    "precision": round(m["precision"], 4),
                    "recall": round(m["recall"], 4),
                    "f1": round(m["f1-score"], 4),
                    "soporte": int(m["support"]),
                })
            filas.append({"modelo": r["modelo"], "clase": "ACCURACY GLOBAL",
                          "precision": "", "recall": "", "f1": round(rep["accuracy"], 4),
                          "soporte": int(rep["macro avg"]["support"])})
        ruta = os.path.join(self.exp_dir, "tabla_mejores.csv")
        self._guardar_csv(filas, ruta)
        print(f"Tabla comparativa de los mejores guardada en: {ruta}")

    @staticmethod
    def _guardar_csv(resultados, ruta):
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(resultados[0]))
            w.writeheader()
            w.writerows(resultados)
