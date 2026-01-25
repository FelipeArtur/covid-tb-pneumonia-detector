"""
Pacote core - Lógica de machine learning.
"""
from .config import CLASS_NAMES, MODEL_PATH, IMG_SIZE
from .predict import predict_image
from .data import load_and_preprocess_image
from .interpret import display_gradcam

__all__ = [
    'CLASS_NAMES',
    'MODEL_PATH', 
    'IMG_SIZE',
    'predict_image',
    'load_and_preprocess_image',
    'display_gradcam'
]
