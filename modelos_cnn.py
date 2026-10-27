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

# Clases del reto 2: el índice es el número de la carpeta en dataset/
# (0=PARE, 1=SIGA, 2=VELOCIDAD_60, 3=ZONA_ESCOLAR, la amarilla de paso).
# La carpeta 4 (SIN_SENAL) es opcional: fotos de la pista y el robot SIN ninguna
# señal. Con ella la red aprende a decir "no hay nada" y el robot se libera del PARE.
CLASES = ["PARE", "SIGA", "VELOCIDAD_60", "ZONA_ESCOLAR", "SIN_SENAL"]

# Resolución de entrada de las CNN (la de los apuntes de clase).
IMG_SIZE = (224, 224)
