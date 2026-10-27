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
# (0=PARE, 1=SIGA, 2=ZONA_ESCOLAR, 3=VELOCIDAD_60).
CLASES = ["PARE", "SIGA", "ZONA_ESCOLAR", "VELOCIDAD_60"]

# Resolución de entrada de las CNN (la de los apuntes de clase).
IMG_SIZE = (224, 224)
