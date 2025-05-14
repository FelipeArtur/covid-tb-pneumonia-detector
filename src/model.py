import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import time

# Diretório base do projeto
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Diretórios de dados
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "TRAIN"
VAL_DIR = DATASET_DIR / "VAL"
TEST_DIR = DATASET_DIR / "TEST"

# Diretório para salvar modelos treinados
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "best_model.h5"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Parâmetros do modelo
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 
NUM_CLASSES = 4

# Nomes das classes (ordem deve ser igual à dos diretórios)
CLASS_NAMES = ["COVID", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

def create_data_generators():
    """
    Cria e retorna os geradores de dados para treino, validação e teste.
    """
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
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode="categorical",
        shuffle=False,
        classes=CLASS_NAMES
    )
    
    return train_gen, val_gen, test_gen

def build_model():
    """
    Constrói e retorna o modelo MobileNetV2 com camadas superiores customizadas.
    """
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Congela as camadas do modelo base
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def plot_history(history):
    """
    Plota e salva o histórico de treino (acurácia e perda).
    """
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(RESULTS_DIR / f'training_history_{timestamp}.png')
    plt.close()

def evaluate_model(model, test_gen):
    """
    Avalia o modelo e exibe métricas de desempenho.
    """
    # Avalia o modelo
    loss, acc = model.evaluate(test_gen)
    print(f"\nTest accuracy: {acc*100:.2f}%")

    # Gera previsões
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    # Exibe relatório de classificação
    print("\nClassification Report:\n")
    report = classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES)
    print(report)
    
    # Salva o relatório de classificação em um arquivo
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(RESULTS_DIR / f'classification_report_{timestamp}.txt', 'w') as f:
        f.write(f"Test accuracy: {acc*100:.2f}%\n\n")
        f.write(report)

    # Exibe a matriz de confusão
    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'confusion_matrix_{timestamp}.png')
    plt.close()

def train_model():
    """
    Treina o modelo, salva o melhor checkpoint e avalia no conjunto de teste.
    """
    train_gen, val_gen, test_gen = create_data_generators()
    model = build_model()
    
    print(f"\nTraining with {train_gen.samples} training images and {val_gen.samples} validation images")
    
    # Configura checkpoints e early stopping
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Treina o modelo
    print("\nTraining model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )

    # Plota o histórico de treino
    plot_history(history)
    
    # Carrega o melhor modelo salvo
    best_model = load_model(MODEL_PATH)
    
    # Avalia o modelo
    evaluate_model(best_model, test_gen)
    
    return best_model

if __name__ == "__main__":
    # Validação dos diretórios de dados
    if not (TRAIN_DIR.exists() and VAL_DIR.exists() and TEST_DIR.exists()):
        print("ERRO: Diretórios do dataset não encontrados. Verifique a estrutura do dataset.")
        print("Estrutura esperada:")
        print(f"dataset/TRAIN/[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]")
        print(f"dataset/VAL/[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]")
        print(f"dataset/TEST/[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]")
        exit(1)
    
    # Treinamento do modelo
    model = train_model()
    print("\nTreinamento concluído. Melhor modelo salvo em:", MODEL_PATH)
