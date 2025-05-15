import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import argparse
from pathlib import Path

# Add the project root directory to Python's path to allow local imports
sys.path.append(str(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Diretório base do projeto
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Caminho do modelo treinado
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.h5"

# Tamanho da imagem (deve ser igual ao usado no treino)
IMG_SIZE = (224, 224)

# Nomes das classes (ordem igual ao treino)
CLASS_NAMES = ["COVID", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

def load_and_preprocess_image(image_path):
    """
    Carrega e pré-processa uma imagem para predição.
    """
    try:
        # Convert string path to Path object for better cross-platform compatibility
        image_path = Path(image_path)
        
        # Check if the file exists
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return None, None
            
        # Check if it's a supported image format
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"Error: Unsupported image format: {image_path.suffix}")
            print("Supported formats: .jpg, .jpeg, .png, .bmp")
            return None, None
            
        img = load_img(str(image_path), target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img, img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

def predict_image(image_path, show_plot=True, show_gradcam=False):
    """
    Prediz a classe de uma imagem de raio-X.
    """
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print(f"Please train the model first using 'python src/model.py'")
        return None

    # Load the trained model
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load and preprocess the image
    img, img_array = load_and_preprocess_image(image_path)
    if img is None or img_array is None:
        return None
    
    # Make prediction
    try:
        predictions = model.predict(img_array, verbose=0)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

    # Create results dictionary
    results = {
        'image_path': image_path,
        'predictions': {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))},
        'predicted_class': CLASS_NAMES[np.argmax(predictions)],
        'confidence': float(np.max(predictions))
    }
    
    # Display results
    if show_plot:
        if show_gradcam:
            # Import here to avoid circular imports
            from src.interpret import display_gradcam
            
            # Generate Grad-CAM components
            gradcam_data = display_gradcam(model, img, img_array, CLASS_NAMES, np.argmax(predictions))
            
            if gradcam_data:
                # Create a figure with a 2x2 grid layout for better organization
                fig, axes = plt.subplots(2, 2, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]})
                
                # Row 1: Original image, heatmap, and overlay
                axes[0, 0].imshow(gradcam_data['original_img'])
                axes[0, 0].set_title("Original X-ray", fontsize=14)
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(gradcam_data['heatmap'], cmap='jet')
                axes[0, 1].set_title(f"Grad-CAM Heatmap", fontsize=14)
                axes[0, 1].set_xlabel(f"Class: {gradcam_data['class_name']} ({results['confidence']*100:.1f}% confidence)")
                axes[0, 1].axis('off')
                
                # Row 2: Overlay image and probability bars
                axes[1, 0].imshow(gradcam_data['overlay_img'])
                axes[1, 0].set_title("Heatmap Overlay", fontsize=14)
                axes[1, 0].set_xlabel("Highlighted areas show regions important for the prediction")
                axes[1, 0].axis('off')
                
                # Probability bars with clearer formatting
                colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
                bars = axes[1, 1].barh(CLASS_NAMES, predictions, color=colors)
                axes[1, 1].set_title('Prediction Probabilities', fontsize=14)
                axes[1, 1].set_xlabel('Probability')
                axes[1, 1].set_xlim(0, 1)
                
                # Add probability values as text with improved formatting
                for i, bar in enumerate(bars):
                    text_color = 'black' if predictions[i] < 0.7 else 'white'
                    weight = 'bold' if i == np.argmax(predictions) else 'normal'
                    axes[1, 1].text(
                        min(bar.get_width() + 0.01, 0.99),
                        bar.get_y() + bar.get_height()/2, 
                        f'{predictions[i]:.1%}',
                        va='center',
                        ha='right' if predictions[i] > 0.9 else 'left',
                        color=text_color,
                        weight=weight
                    )
                
                # Add a title to the entire figure
                fig.suptitle(
                    f"X-ray Analysis: {Path(image_path).name}\n"
                    f"Prediction: {results['predicted_class']} (Confidence: {results['confidence']*100:.1f}%)",
                    fontsize=16
                )
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)  # Make room for the title
                plt.show()
        else:
            # Regular visualization without Grad-CAM
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Display the image with proper title
            axes[0].imshow(img)
            axes[0].set_title("X-ray Image", fontsize=14)
            axes[0].set_xlabel(f"File: {Path(image_path).name}")
            axes[0].axis('off')
            
            # Display prediction probabilities with improved formatting
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
            bars = axes[1].barh(CLASS_NAMES, predictions, color=colors)
            axes[1].set_title('Prediction Probabilities', fontsize=14)
            axes[1].set_xlabel('Probability')
            axes[1].set_xlim(0, 1)
            
            # Highlight the predicted class
            for i, bar in enumerate(bars):
                text_color = 'black' if predictions[i] < 0.7 else 'white'
                weight = 'bold' if i == np.argmax(predictions) else 'normal'
                axes[1].text(
                    min(bar.get_width() + 0.01, 0.99),
                    bar.get_y() + bar.get_height()/2, 
                    f'{predictions[i]:.1%}',
                    va='center',
                    ha='right' if predictions[i] > 0.9 else 'left',
                    color=text_color,
                    weight=weight
                )
            
            # Add a title to the entire figure
            fig.suptitle(
                f"Predicted condition: {results['predicted_class']} "
                f"(Confidence: {results['confidence']*100:.1f}%)",
                fontsize=16
            )
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for the title
            plt.show()
    
    # Print results
    print("\n=== Prediction Results ===")
    print(f"Image: {Path(image_path).name}")
    print("\nProbabilities:")
    for class_name, prob in results['predictions'].items():
        print(f"{class_name}: {prob*100:.2f}%")
    
    # Get the predicted class
    print(f"\nPredicted condition: {results['predicted_class']} (Confidence: {results['confidence']*100:.2f}%)")
    
    return results

def main():
    """
    Interface de linha de comando para predição de uma imagem.
    """
    parser = argparse.ArgumentParser(description='Predict chest X-ray condition.')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the X-ray image to predict')
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not display the plot (useful for batch processing)')
    parser.add_argument('--gradcam', action='store_true',
                        help='Show Grad-CAM visualization to highlight important regions')
    
    args = parser.parse_args()
    
    # Predict the image
    predict_image(args.image, show_plot=not args.no_plot, show_gradcam=args.gradcam)

if __name__ == "__main__":
    main()
