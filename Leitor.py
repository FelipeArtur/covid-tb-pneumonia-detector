import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import load_img, img_to_array

# Caminhos
BASE_DIR = "C:/Ragnarok/covid-tb-pneumonia-detector/dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "TRAIN")
VAL_DIR = os.path.join(BASE_DIR, "VAL")
TEST_DIR = os.path.join(BASE_DIR, "TEST")
MODEL_PATH = "C:/Ragnarok/covid-tb-pneumonia-detector/BESTMODEL/best_model.h5"

# Parâmetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 2
NUM_CLASSES = 4

# Preprocessamento
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_gen.class_indices.keys())

def treinar_modelo():
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    plot_history(history)

    return model

def plot_history(history):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()

    plt.tight_layout()
    plt.show()

def carregar_modelo():
    return load_model(MODEL_PATH)

def prever_imagem(caminho_imagem):
    model = carregar_modelo()
    img = load_img(caminho_imagem, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0]

    plt.imshow(img)
    plt.axis('off')
    plt.title("Imagem analisada")
    plt.show()

    print("Probabilidades:")
    for i, prob in enumerate(pred):
        print(f"{class_names[i]}: {prob*100:.2f}%")

# Evita que o código treine ao ser importado
if __name__ == "__main__":
    model = treinar_modelo()
    print("\nModelo treinado e salvo com sucesso.")

    loss, acc = model.evaluate(test_gen)
    print(f"\nAcurácia final no teste: {acc*100:.2f}%")

    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    print("\nRelatório de Classificação:\n")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.show()
