"""
Módulo responsável pelo treinamento e avaliação do modelo.
"""
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

from .config import (
    CLASS_NAMES, MODEL_PATH, RESULTS_DIR,
    TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE
)
from .data import create_data_generators
from .model import build_model, AVAILABLE_MODELS


def calculate_class_weights(train_gen):
    """
    Calcula pesos das classes para lidar com desbalanceamento.
    
    Args:
        train_gen: Gerador de dados de treino
    
    Returns:
        Dicionário com pesos por classe
    """
    class_counts = np.bincount(train_gen.classes)
    print(f"Contagem por classe: {list(zip(CLASS_NAMES, class_counts))}")
    
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Pesos das classes: {class_weights_dict}")
    
    return class_weights_dict


def plot_history(history):
    """
    Plota e salva o histórico de treino (acurácia e perda).
    """
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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(RESULTS_DIR / f'training_history_{timestamp}.png')
    plt.close()


def evaluate_model(model, test_gen):
    """
    Avalia o modelo e exibe métricas de desempenho.
    """
    loss, acc = model.evaluate(test_gen)
    print(f"\nAcurácia no teste: {acc*100:.2f}%")

    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    print("\nRelatório de Classificação:\n")
    report = classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES)
    print(report)
    
    # Salvar relatório
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(RESULTS_DIR / f'classification_report_{timestamp}.txt', 'w') as f:
        f.write(f"Acurácia no teste: {acc*100:.2f}%\n\n")
        f.write(report)

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'confusion_matrix_{timestamp}.png')
    plt.close()
    
    # Métricas por classe
    print("\nDesempenho por Classe:")
    for i, class_name in enumerate(CLASS_NAMES):
        true_pos = cm[i, i]
        false_neg = np.sum(cm[i, :]) - true_pos
        false_pos = np.sum(cm[:, i]) - true_pos
        true_neg = np.sum(cm) - true_pos - false_neg - false_pos
        
        sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        
        print(f"{class_name}:")
        print(f"  - Sensibilidade/Recall: {sensitivity:.4f}")
        print(f"  - Especificidade: {specificity:.4f}")
        print(f"  - Precisão: {precision:.4f}")
        
        with open(RESULTS_DIR / f'classification_report_{timestamp}.txt', 'a') as f:
            f.write(f"\n{class_name}:\n")
            f.write(f"  - Sensibilidade/Recall: {sensitivity:.4f}\n")
            f.write(f"  - Especificidade: {specificity:.4f}\n")
            f.write(f"  - Precisão: {precision:.4f}\n")


def train_model(base_model_name="mobilenetv2", img_size=IMG_SIZE, batch_size=BATCH_SIZE, 
                epochs=5, dropout_rate=0.3, fine_tuning=True, fine_tuning_layers=30,
                use_class_weights=True, enhanced_aug=True):
    """
    Treina o modelo, salva o melhor checkpoint e avalia no conjunto de teste.
    
    Args:
        base_model_name: Arquitetura backbone
        img_size: Dimensões da imagem de entrada
        batch_size: Tamanho do batch
        epochs: Número de épocas
        dropout_rate: Taxa de dropout
        fine_tuning: Se deve usar fine-tuning
        fine_tuning_layers: Número de camadas para fine-tuning
        use_class_weights: Se deve usar pesos de classe
        enhanced_aug: Se deve usar augmentation avançado
        
    Returns:
        Modelo treinado
    """
    train_gen, val_gen, test_gen = create_data_generators(img_size, batch_size, enhanced_aug)
    
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(train_gen)
    
    model = build_model(
        base_model_name=base_model_name,
        img_size=img_size,
        dropout_rate=dropout_rate,
        fine_tuning=fine_tuning,
        fine_tuning_layers=fine_tuning_layers
    )
    
    print(f"\nTreinando com {train_gen.samples} imagens de treino e {val_gen.samples} de validação")
    
    # Callbacks
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
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Salvar parâmetros
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
    
    print("\nTreinando modelo...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights
    )

    plot_history(history)
    
    best_model = load_model(MODEL_PATH)
    evaluate_model(best_model, test_gen)
    
    return best_model


def parse_arguments():
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Treinar modelo de classificação de raio-X")
    parser.add_argument("--model", type=str, default="mobilenetv2",
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Arquitetura do modelo base")
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224],
                        help="Dimensões da imagem (altura, largura)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Tamanho do batch")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Número de épocas")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Taxa de dropout")
    parser.add_argument("--no-fine-tuning", action="store_true",
                        help="Desabilitar fine-tuning")
    parser.add_argument("--fine-tuning-layers", type=int, default=30,
                        help="Número de camadas para fine-tuning")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Desabilitar pesos de classe")
    parser.add_argument("--no-enhanced-aug", action="store_true",
                        help="Desabilitar augmentation avançado")
    
    return parser.parse_args()


def main():
    """Ponto de entrada para treinamento via CLI."""
    args = parse_arguments()
    
    # Validação dos diretórios
    if not (TRAIN_DIR.exists() and VAL_DIR.exists() and TEST_DIR.exists()):
        print("ERRO: Diretórios do dataset não encontrados.")
        print("Estrutura esperada:")
        print(f"  {TRAIN_DIR}")
        print(f"  {VAL_DIR}")
        print(f"  {TEST_DIR}")
        exit(1)
    
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
    
    print(f"\nTreinamento concluído. Melhor modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
