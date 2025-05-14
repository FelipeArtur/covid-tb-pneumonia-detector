import os
import sys
import pandas as pd
from pathlib import Path
import argparse
import glob
import matplotlib.pyplot as plt

# Add src directory to Python path
sys.path.append(str(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import prediction function
from src.predict import predict_image

def predict_batch(directory_path, output_file=None, extensions=None, save_gradcam=False):
    """
    Predict all images in a directory and output results to a CSV file.
    
    Args:
        directory_path (str): Path to directory containing images
        output_file (str): Path to output CSV file
        extensions (list): List of file extensions to include (default: ['.png', '.jpg', '.jpeg'])
        save_gradcam (bool): Whether to save Grad-CAM visualizations for each image
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg']
    
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory_path} is not a valid directory")
        return
    
    # Create list of image files
    image_files = []
    for ext in extensions:
        image_files.extend(list(directory.glob(f"*{ext}")))
        image_files.extend(list(directory.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No images found in {directory_path} with extensions {extensions}")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    
    # Create output directory for Grad-CAM images if needed
    if save_gradcam:
        gradcam_dir = Path(directory) / "gradcam_results"
        gradcam_dir.mkdir(exist_ok=True)
        print(f"Grad-CAM visualizations will be saved to {gradcam_dir}")
        
        # Import here to avoid circular imports
        from src.interpret import display_gradcam
        from tensorflow.keras.models import load_model
        
        # Load model once for all predictions
        model_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "models" / "best_model.h5"
        try:
            model = load_model(str(model_path))
            class_names = ["COVID", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    # Process each image
    results = []
    for i, img_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {img_path.name}")
        
        # Basic prediction without showing plot
        result = predict_image(str(img_path), show_plot=False)
        
        if result:
            result_data = {
                'filename': img_path.name,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence']
            }
            # Add individual class probabilities
            for class_name, prob in result['predictions'].items():
                result_data[f'{class_name}_probability'] = prob
                
            results.append(result_data)
            
            # Generate and save Grad-CAM visualization if requested
            if save_gradcam:
                from src.predict import load_and_preprocess_image
                
                img, img_array = load_and_preprocess_image(str(img_path))
                if img is not None and img_array is not None:
                    pred_idx = list(class_names).index(result['predicted_class'])
                    fig = display_gradcam(model, img, img_array, class_names, pred_idx)
                    
                    if fig:
                        save_path = gradcam_dir / f"{img_path.stem}_gradcam.png"
                        fig.savefig(save_path, dpi=200, bbox_inches='tight')
                        plt.close(fig)
                        print(f"  Saved Grad-CAM to {save_path}")
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save to CSV if output file specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total images processed: {len(results)}")
    print("Predictions by class:")
    class_counts = df['predicted_class'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(results)*100:.1f}%)")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Batch predict chest X-ray images in a directory.')
    parser.add_argument('--dir', type=str, required=True,
                      help='Directory containing X-ray images to predict')
    parser.add_argument('--output', type=str, default=None,
                      help='Output CSV file path (optional)')
    parser.add_argument('--ext', type=str, nargs='+', default=['.png', '.jpg', '.jpeg'],
                      help='File extensions to process (default: .png .jpg .jpeg)')
    parser.add_argument('--save-gradcam', action='store_true',
                      help='Generate and save Grad-CAM visualizations for each image')
    
    args = parser.parse_args()
    
    # Run batch prediction
    predict_batch(args.dir, args.output, args.ext, args.save_gradcam)

if __name__ == "__main__":
    main()
