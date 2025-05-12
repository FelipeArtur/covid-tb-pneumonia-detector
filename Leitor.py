import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Caminho para os diretórios organizados por classe (uma pasta por classe com imagens dentro)
base_dir = 'dataset'  # Ex: dataset/NORMAL/, dataset/COVID/, etc.

# Pré-processamento com aumento de dados
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Modelo base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # 4 classes

model = Model(inputs=base_model.input, outputs=predictions)

# Congelar camadas do MobileNetV2
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento
model.fit(train_data, validation_data=val_data, epochs=10)

# Para prever a probabilidade de cada classe
import numpy as np
from tensorflow.keras.preprocessing import image

img_path = 'teste.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

preds = model.predict(x)
class_labels = list(train_data.class_indices.keys())
for label, prob in zip(class_labels, preds[0]):
    print(f"{label}: {prob*100:.2f}%")
