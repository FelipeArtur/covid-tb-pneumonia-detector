"""
Módulo responsável pela construção e compilação do modelo.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, EfficientNetB0, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

from .config import CLASS_NAMES, IMG_SIZE

# Modelos disponíveis
AVAILABLE_MODELS = {
    "mobilenetv2": MobileNetV2,
    "resnet50v2": ResNet50V2,
    "efficientnetb0": EfficientNetB0,
    "inceptionv3": InceptionV3
}


def build_model(base_model_name="mobilenetv2", img_size=IMG_SIZE, 
                dropout_rate=0.3, fine_tuning=True, fine_tuning_layers=30):
    """
    Constrói e retorna o modelo compilado.
    
    Args:
        base_model_name: Nome do backbone (mobilenetv2, resnet50v2, etc.)
        img_size: Tuple com dimensões da imagem
        dropout_rate: Taxa de dropout para regularização
        fine_tuning: Se deve usar fine-tuning
        fine_tuning_layers: Número de camadas do topo para descongelar
    
    Returns:
        Modelo Keras compilado
    """
    base_model_name = base_model_name.lower()
    if base_model_name not in AVAILABLE_MODELS:
        print(f"Modelo {base_model_name} não disponível. Usando MobileNetV2.")
        base_model_name = "mobilenetv2"
    
    base_model_constructor = AVAILABLE_MODELS[base_model_name]
    
    # InceptionV3 requer tamanho mínimo de 75x75
    if base_model_name == "inceptionv3" and (img_size[0] < 75 or img_size[1] < 75):
        img_size = (299, 299)
        print(f"Tamanho ajustado para {img_size} para InceptionV3")
    
    # Criar modelo base
    base_model = base_model_constructor(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet"
    )
    
    # Adicionar camadas customizadas
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout_rate)(x)
    output = Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    # Congelar todas as camadas base inicialmente
    for layer in base_model.layers:
        layer.trainable = False
    
    # Fine-tuning: descongelar algumas camadas do topo
    if fine_tuning:
        for layer in base_model.layers[-fine_tuning_layers:]:
            layer.trainable = True
    
    # Contar parâmetros
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"Parâmetros treináveis: {trainable_count:,}")
    print(f"Parâmetros não-treináveis: {non_trainable_count:,}")
    
    # Learning rate menor para fine-tuning
    lr = 0.0001 if fine_tuning else 0.001
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
