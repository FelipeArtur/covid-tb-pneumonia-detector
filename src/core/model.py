import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, EfficientNetB0, InceptionV3
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from pathlib import Path
import time
import argparse

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

# Nomes das classes (ordem deve ser igual à dos diretórios)
CLASS_NAMES = ["COVID", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

# Modelos disponíveis
AVAILABLE_MODELS = {
    "mobilenetv2": MobileNetV2,
    "resnet50v2": ResNet50V2,
    "efficientnetb0": EfficientNetB0,
    "inceptionv3": InceptionV3
}

def create_data_generators(img_size=(224, 224), batch_size=32, enhanced_aug=False):
    """
    Cria e retorna os geradores de dados para treino, validação e teste.
    
    Args:
        img_size: Tuple with image dimensions
        batch_size: Batch size for training
        enhanced_aug: Whether to use enhanced augmentation
    """
    if enhanced_aug:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=False,  # X-rays should maintain orientation
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

def calculate_class_weights(train_gen):
    """
    Calculates class weights to handle class imbalance.
    
    Args:
        train_gen: Training data generator
    
    Returns:
        Dictionary of class weights
    """
    # Get class distribution
    class_counts = np.bincount(train_gen.classes)
    print(f"Class counts: {list(zip(CLASS_NAMES, class_counts))}")
    
    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    
    # Convert to dictionary
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weights_dict}")
    
    return class_weights_dict

def build_model(base_model_name="mobilenetv2", img_size=(224, 224), 
                dropout_rate=0.3, fine_tuning=True, fine_tuning_layers=30):
    """
    Constrói e retorna o modelo com diferentes opções de backbone e fine-tuning.
    
    Args:
        base_model_name: Name of the backbone model (mobilenetv2, resnet50v2, etc.)
        img_size: Tuple with image dimensions
        dropout_rate: Dropout rate for regularization
        fine_tuning: Whether to use fine-tuning
        fine_tuning_layers: Number of top layers to unfreeze for fine-tuning
    
    Returns:
        Compiled Keras model
    """
    # Check if the selected model is available
    base_model_name = base_model_name.lower()
    if base_model_name not in AVAILABLE_MODELS:
        print(f"Model {base_model_name} not available. Using MobileNetV2 instead.")
        base_model_name = "mobilenetv2"
    
    # Get the base model constructor
    base_model_constructor = AVAILABLE_MODELS[base_model_name]
    
    # Adjust input size for InceptionV3 which requires minimum 75x75
    if base_model_name == "inceptionv3" and (img_size[0] < 75 or img_size[1] < 75):
        img_size = (299, 299)
        print(f"Adjusted input size to {img_size} for InceptionV3")
    
    # Create base model
    base_model = base_model_constructor(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet"
    )
    
    # Add custom top layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout_rate)(x)
    output = Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    # Initially freeze all base layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # If fine-tuning is enabled, unfreeze some top layers
    if fine_tuning:
        # Unfreeze the top N layers
        for layer in base_model.layers[-fine_tuning_layers:]:
            layer.trainable = True
    
    # Count trainable and non-trainable parameters
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    
    # Use a lower learning rate for fine-tuning
    lr = 0.0001 if fine_tuning else 0.001
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
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
    
    # Calculate and display per-class metrics
    print("\nPer-Class Performance:")
    for i, class_name in enumerate(CLASS_NAMES):
        true_pos = cm[i, i]
        false_neg = np.sum(cm[i, :]) - true_pos
        false_pos = np.sum(cm[:, i]) - true_pos
        true_neg = np.sum(cm) - true_pos - false_neg - false_pos
        
        sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        
        print(f"{class_name}:")
        print(f"  - Sensitivity/Recall: {sensitivity:.4f}")
        print(f"  - Specificity: {specificity:.4f}")
        print(f"  - Precision: {precision:.4f}")
        
        # Save these metrics to the report file
        with open(RESULTS_DIR / f'classification_report_{timestamp}.txt', 'a') as f:
            f.write(f"\n{class_name}:\n")
            f.write(f"  - Sensitivity/Recall: {sensitivity:.4f}\n")
            f.write(f"  - Specificity: {specificity:.4f}\n")
            f.write(f"  - Precision: {precision:.4f}\n")

def train_model(base_model_name="mobilenetv2", img_size=(224, 224), batch_size=32, 
                epochs=5, dropout_rate=0.3, fine_tuning=True, fine_tuning_layers=30,
                use_class_weights=True, enhanced_aug=True):
    """
    Treina o modelo, salva o melhor checkpoint e avalia no conjunto de teste.
    
    Args:
        base_model_name: The backbone architecture to use
        img_size: Input image dimensions
        batch_size: Training batch size
        epochs: Number of epochs for training
        dropout_rate: Dropout rate for regularization
        fine_tuning: Whether to fine-tune the base model
        fine_tuning_layers: Number of layers to fine-tune
        use_class_weights: Whether to use class weights
        enhanced_aug: Whether to use enhanced data augmentation
        
    Returns:
        Trained model
    """
    train_gen, val_gen, test_gen = create_data_generators(img_size, batch_size, enhanced_aug)
    
    # Calculate class weights if needed
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(train_gen)
    
    # Build and compile the model
    model = build_model(
        base_model_name=base_model_name,
        img_size=img_size,
        dropout_rate=dropout_rate,
        fine_tuning=fine_tuning,
        fine_tuning_layers=fine_tuning_layers
    )
    
    print(f"\nTraining with {train_gen.samples} training images and {val_gen.samples} validation images")
    
    # Configure callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Add learning rate reduction on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Record model parameters for documentation
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_params = {
        "base_model": base_model_name,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "dropout_rate": dropout_rate,
        "fine_tuning": fine_tuning,
        "fine_tuning_layers": fine_tuning_layers if fine_tuning else 0,
        "use_class_weights": use_class_weights,
        "enhanced_augmentation": enhanced_aug,
        "timestamp": timestamp
    }
    
    with open(RESULTS_DIR / f'model_params_{timestamp}.txt', 'w') as f:
        for key, value in model_params.items():
            f.write(f"{key}: {value}\n")
    
    # Treina o modelo
    print("\nTraining model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights
    )

    # Plota o histórico de treino
    plot_history(history)
    
    # Carrega o melhor modelo salvo
    best_model = load_model(MODEL_PATH)
    
    # Avalia o modelo
    evaluate_model(best_model, test_gen)
    
    return best_model

def parse_arguments():
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train X-ray classification model")
    parser.add_argument("--model", type=str, default="mobilenetv2",
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Base model architecture")
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224],
                        help="Image dimensions (height, width)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate for regularization")
    parser.add_argument("--no-fine-tuning", action="store_true",
                        help="Disable fine-tuning (freeze all base layers)")
    parser.add_argument("--fine-tuning-layers", type=int, default=30,
                        help="Number of top layers to unfreeze for fine-tuning")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class weights")
    parser.add_argument("--no-enhanced-aug", action="store_true",
                        help="Disable enhanced data augmentation")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Validação dos diretórios de dados
    if not (TRAIN_DIR.exists() and VAL_DIR.exists() and TEST_DIR.exists()):
        print("ERRO: Diretórios do dataset não encontrados. Verifique a estrutura do dataset.")
        print("Estrutura esperada:")
        print(f"dataset/TRAIN/[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]")
        print(f"dataset/VAL/[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]")
        print(f"dataset/TEST/[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]")
        exit(1)
    
    # Treinamento do modelo com os parâmetros especificados
    model = train_model(
        base_model_name=args.model,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        dropout_rate=args.dropout,
        fine_tuning=not args.no_fine_tuning,
        fine_tuning_layers=args.fine_tuning_layers,
        use_class_weights=not args.no_class_weights,
        enhanced_aug=not args.no_enhanced_aug
    )
    
    print("\nTreinamento concluído. Melhor modelo salvo em:", MODEL_PATH)
