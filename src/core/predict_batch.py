"""
Módulo responsável pela predição em lote de imagens.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model

from .config import CLASS_NAMES, MODEL_PATH, SUPPORTED_EXTENSIONS
from .predict import predict_image
from .data import load_and_preprocess_image
from .interpret import display_gradcam


def predict_batch(directory_path, output_file=None, extensions=None, save_gradcam=False):
    """
    Prediz todas as imagens em um diretório.
    
    Args:
        directory_path: Caminho para o diretório com imagens
        output_file: Caminho para arquivo CSV de saída (opcional)
        extensions: Lista de extensões a processar
        save_gradcam: Se deve salvar visualizações Grad-CAM
        
    Returns:
        DataFrame com resultados ou None se houver erro
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS
    
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"Erro: {directory_path} não é um diretório válido")
        return None
    
    # Coletar arquivos de imagem
    image_files = []
    for ext in extensions:
        image_files.extend(list(directory.glob(f"*{ext}")))
        image_files.extend(list(directory.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"Nenhuma imagem encontrada em {directory_path}")
        return None
    
    print(f"Encontradas {len(image_files)} imagens. Processando...")
    
    # Carregar modelo para Grad-CAM se necessário
    model = None
    gradcam_dir = None
    if save_gradcam:
        gradcam_dir = directory / "gradcam_results"
        gradcam_dir.mkdir(exist_ok=True)
        print(f"Grad-CAM será salvo em {gradcam_dir}")
        
        try:
            model = load_model(str(MODEL_PATH))
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return None
    
    # Processar imagens
    results = []
    for i, img_path in enumerate(image_files):
        print(f"Processando {i+1}/{len(image_files)}: {img_path.name}")
        
        result = predict_image(str(img_path), show_plot=False)
        
        if result:
            result_data = {
                'filename': img_path.name,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence']
            }
            for class_name, prob in result['predictions'].items():
                result_data[f'{class_name}_probability'] = prob
                
            results.append(result_data)
            
            # Gerar Grad-CAM se solicitado
            if save_gradcam and model:
                _save_gradcam(model, img_path, result, gradcam_dir)
    
    df = pd.DataFrame(results)
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Resultados salvos em {output_file}")
    
    _print_summary(df, results)
    
    return df


def _save_gradcam(model, img_path, result, gradcam_dir):
    """Salva visualização Grad-CAM para uma imagem."""
    img, img_array = load_and_preprocess_image(str(img_path))
    if img is not None and img_array is not None:
        pred_idx = CLASS_NAMES.index(result['predicted_class'])
        gradcam_result = display_gradcam(model, img, img_array, CLASS_NAMES, pred_idx)
        
        if gradcam_result and 'overlay_img' in gradcam_result:
            save_path = gradcam_dir / f"{img_path.stem}_gradcam.png"
            gradcam_result['overlay_img'].save(save_path)
            print(f"  Grad-CAM salvo: {save_path.name}")


def _print_summary(df, results):
    """Imprime resumo das predições."""
    print("\n=== Resumo ===")
    print(f"Total processadas: {len(results)}")
    print("Predições por classe:")
    class_counts = df['predicted_class'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(results)*100:.1f}%)")


def main():
    """Interface de linha de comando para predição em lote."""
    parser = argparse.ArgumentParser(description='Predição em lote de raios-X.')
    parser.add_argument('--dir', type=str, required=True,
                        help='Diretório com imagens')
    parser.add_argument('--output', type=str, default=None,
                        help='Arquivo CSV de saída')
    parser.add_argument('--ext', type=str, nargs='+', default=SUPPORTED_EXTENSIONS,
                        help='Extensões a processar')
    parser.add_argument('--save-gradcam', action='store_true',
                        help='Salvar visualizações Grad-CAM')
    
    args = parser.parse_args()
    predict_batch(args.dir, args.output, args.ext, args.save_gradcam)


if __name__ == "__main__":
    main()
