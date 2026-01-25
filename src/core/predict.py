"""
Módulo responsável pela predição de imagens individuais.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model

from .config import CLASS_NAMES, MODEL_PATH, IMG_SIZE
from .data import load_and_preprocess_image


def predict_image(image_path, show_plot=True, show_gradcam=False):
    """
    Prediz a classe de uma imagem de raio-X.
    
    Args:
        image_path: Caminho para a imagem
        show_plot: Se deve exibir visualização
        show_gradcam: Se deve mostrar Grad-CAM
        
    Returns:
        Dicionário com resultados ou None se houver erro
    """
    if not MODEL_PATH.exists():
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
        print("Execute o treinamento primeiro: python -m src.core.train")
        return None

    try:
        model = load_model(str(MODEL_PATH))
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None

    img, img_array = load_and_preprocess_image(image_path)
    if img is None or img_array is None:
        return None
    
    try:
        predictions = model.predict(img_array, verbose=0)[0]
    except Exception as e:
        print(f"Erro durante predição: {e}")
        return None

    results = {
        'image_path': str(image_path),
        'predictions': {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))},
        'predicted_class': CLASS_NAMES[np.argmax(predictions)],
        'confidence': float(np.max(predictions))
    }
    
    if show_plot:
        _display_results(img, image_path, predictions, results, model, img_array, show_gradcam)
    
    _print_results(image_path, results)
    
    return results


def _display_results(img, image_path, predictions, results, model, img_array, show_gradcam):
    """Exibe resultados visualmente."""
    if show_gradcam:
        from .interpret import display_gradcam
        
        gradcam_data = display_gradcam(model, img, img_array, CLASS_NAMES, np.argmax(predictions))
        
        if gradcam_data:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]})
            
            axes[0, 0].imshow(gradcam_data['original_img'])
            axes[0, 0].set_title("Raio-X Original", fontsize=14)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(gradcam_data['heatmap'], cmap='jet')
            axes[0, 1].set_title("Heatmap Grad-CAM", fontsize=14)
            axes[0, 1].set_xlabel(f"Classe: {gradcam_data['class_name']} ({results['confidence']*100:.1f}%)")
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(gradcam_data['overlay_img'])
            axes[1, 0].set_title("Sobreposição", fontsize=14)
            axes[1, 0].set_xlabel("Áreas destacadas são importantes para a predição")
            axes[1, 0].axis('off')
            
            _plot_probabilities(axes[1, 1], predictions)
            
            fig.suptitle(
                f"Análise de Raio-X: {Path(image_path).name}\n"
                f"Predição: {results['predicted_class']} ({results['confidence']*100:.1f}%)",
                fontsize=16
            )
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()
            return
    
    # Visualização sem Grad-CAM
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(img)
    axes[0].set_title("Imagem de Raio-X", fontsize=14)
    axes[0].set_xlabel(f"Arquivo: {Path(image_path).name}")
    axes[0].axis('off')
    
    _plot_probabilities(axes[1], predictions)
    
    fig.suptitle(
        f"Condição Predita: {results['predicted_class']} "
        f"({results['confidence']*100:.1f}%)",
        fontsize=16
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def _plot_probabilities(ax, predictions):
    """Plota barras de probabilidade."""
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    bars = ax.barh(CLASS_NAMES, predictions, color=colors)
    ax.set_title('Probabilidades', fontsize=14)
    ax.set_xlabel('Probabilidade')
    ax.set_xlim(0, 1)
    
    for i, bar in enumerate(bars):
        text_color = 'black' if predictions[i] < 0.7 else 'white'
        weight = 'bold' if i == np.argmax(predictions) else 'normal'
        ax.text(
            min(bar.get_width() + 0.01, 0.99),
            bar.get_y() + bar.get_height()/2, 
            f'{predictions[i]:.1%}',
            va='center',
            ha='right' if predictions[i] > 0.9 else 'left',
            color=text_color,
            weight=weight
        )


def _print_results(image_path, results):
    """Imprime resultados no console."""
    print("\n=== Resultados da Predição ===")
    print(f"Imagem: {Path(image_path).name}")
    print("\nProbabilidades:")
    for class_name, prob in results['predictions'].items():
        print(f"  {class_name}: {prob*100:.2f}%")
    print(f"\nCondição Predita: {results['predicted_class']} ({results['confidence']*100:.2f}%)")


def main():
    """Interface de linha de comando para predição."""
    parser = argparse.ArgumentParser(description='Predizer condição de raio-X torácico.')
    parser.add_argument('--image', type=str, required=True,
                        help='Caminho para a imagem')
    parser.add_argument('--no-plot', action='store_true',
                        help='Não exibir visualização')
    parser.add_argument('--gradcam', action='store_true',
                        help='Mostrar visualização Grad-CAM')
    
    args = parser.parse_args()
    predict_image(args.image, show_plot=not args.no_plot, show_gradcam=args.gradcam)


if __name__ == "__main__":
    main()
