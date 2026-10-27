from keras import applications

# nombre -> (constructor, función de preprocesamiento)
# Mismas arquitecturas trabajadas en clase (5_cnns).
MODELOS = {
    "VGG16": (applications.VGG16, applications.vgg16.preprocess_input),
    "VGG19": (applications.VGG19, applications.vgg19.preprocess_input),
    "MobileNet": (applications.MobileNet, applications.mobilenet.preprocess_input),
    "ResNet50": (applications.ResNet50, applications.resnet50.preprocess_input),
    "DenseNet121": (applications.DenseNet121, applications.densenet.preprocess_input),
}

# Fotos de las señales: una carpeta por clase (0, 1, 2, 3).
DATASET_DIR = "5_cnns/dataset"

# Clases del reto 2: el índice es el número de la carpeta en DATASET_DIR.
# OJO: en las fotos que tomamos la carpeta 2 es la de 60 km/h y la 3 la de
# zona escolar (al revés de la guía del profesor).
CLASES = ["PARE", "SIGA", "VELOCIDAD_60", "ZONA_ESCOLAR"]

# Resolución de entrada de las CNN (la de los apuntes de clase).
IMG_SIZE = (224, 224)
