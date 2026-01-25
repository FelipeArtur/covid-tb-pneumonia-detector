"""
Configurações centralizadas do projeto.
"""
from pathlib import Path

# Diretório base do projeto (raiz)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Diretórios de dados
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "TRAIN"
VAL_DIR = DATASET_DIR / "VAL"
TEST_DIR = DATASET_DIR / "TEST"

# Diretório para salvar modelos treinados
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "best_model.h5"

# Diretório para resultados
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Configurações de imagem
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Nomes das classes (ordem deve ser igual à dos diretórios)
CLASS_NAMES = ["COVID", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

# Extensões de imagem suportadas
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
