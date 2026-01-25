"""
Módulo responsável pelo carregamento e preprocessamento de dados.
"""
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from pathlib import Path

from .config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASS_NAMES, IMG_SIZE, BATCH_SIZE, SUPPORTED_EXTENSIONS
)


def create_data_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE, enhanced_aug=False):
    """
    Cria e retorna os geradores de dados para treino, validação e teste.
    
    Args:
        img_size: Tuple com dimensões da imagem
        batch_size: Tamanho do batch para treinamento
        enhanced_aug: Se deve usar augmentation avançado
        
    Returns:
        Tuple (train_gen, val_gen, test_gen)
    """
    if enhanced_aug:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=False,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            brightness_range=[0.9, 1.1],
            fill_mode='constant',
            cval=0
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            zoom_range=0.1,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=img_size,
        batch_size=1,
        class_mode="categorical",
        shuffle=False,
        classes=CLASS_NAMES
    )
    
    return train_gen, val_gen, test_gen


def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    """
    Carrega e pré-processa uma imagem para predição.
    
    Args:
        image_path: Caminho para a imagem
        img_size: Tamanho alvo da imagem
        
    Returns:
        Tuple (img_original, img_array) ou (None, None) se houver erro
    """
    try:
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"Erro: Arquivo não encontrado: {image_path}")
            return None, None
            
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Erro: Formato não suportado: {image_path.suffix}")
            print(f"Formatos suportados: {SUPPORTED_EXTENSIONS}")
            return None, None
            
        img = load_img(str(image_path), target_size=img_size, color_mode='rgb')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img, img_array
        
    except Exception as e:
        print(f"Erro ao carregar imagem: {e}")
        return None, None
