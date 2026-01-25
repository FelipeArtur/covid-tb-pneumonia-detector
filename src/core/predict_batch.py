import os
import sys
import pandas as pd
from pathlib import Path
import argparse
import glob
import matplotlib.pyplot as plt

# Adiciona o diretório src ao path do Python
sys.path.append(str(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Importa função de predição
from src.predict import predict_image

def predict_batch(directory_path, output_file=None, extensions=None, save_gradcam=False):
    """
    Prediz todas as imagens em um diretório e salva os resultados em CSV.
    
    Args:
        directory_path (str): Caminho para o diretório contendo as imagens
        output_file (str): Caminho para o arquivo CSV de saída
        extensions (list): Lista de extensões de arquivos a serem incluídas (padrão: ['.png', '.jpg', '.jpeg'])
        save_gradcam (bool): Se deve salvar visualizações Grad-CAM para cada imagem
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg']
    
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"Erro: {directory_path} não é um diretório válido")
        return
    
    # Cria lista de arquivos de imagem
    image_files = []
    for ext in extensions:
        image_files.extend(list(directory.glob(f"*{ext}")))
        image_files.extend(list(directory.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"Nenhuma imagem encontrada em {directory_path} com extensões {extensions}")
        return
    
    print(f"Encontradas {len(image_files)} imagens. Processando...")
    
    # Cria diretório de saída para imagens Grad-CAM, se necessário
    if save_gradcam:
        gradcam_dir = Path(directory) / "gradcam_results"
        gradcam_dir.mkdir(exist_ok=True)
        print(f"Visualizações Grad-CAM serão salvas em {gradcam_dir}")
        
        # Importa aqui para evitar importações circulares
        from src.interpret import display_gradcam
        from tensorflow.keras.models import load_model
        
        # Carrega o modelo uma vez para todas as predições
        model_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "models" / "best_model.h5"
        try:
            model = load_model(str(model_path))
            class_names = ["COVID", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            return None
    
    # Processa cada imagem
    results = []
    for i, img_path in enumerate(image_files):
        print(f"Processando imagem {i+1}/{len(image_files)}: {img_path.name}")
        
        # Predição básica sem exibir plot
        result = predict_image(str(img_path), show_plot=False)
        
        if result:
            result_data = {
                'filename': img_path.name,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence']
            }
            # Adiciona probabilidades individuais de cada classe
            for class_name, prob in result['predictions'].items():
                result_data[f'{class_name}_probability'] = prob
                
            results.append(result_data)
            
            # Gera e salva visualização Grad-CAM, se solicitado
            if save_gradcam:
                # Importa aqui para evitar importações circulares
                from src.interpret import display_gradcam
                from src.predict import load_and_preprocess_image
                
                img, img_array = load_and_preprocess_image(str(img_path))
                if img is not None and img_array is not None:
                    pred_idx = list(class_names).index(result['predicted_class'])
                    fig = display_gradcam(model, img, img_array, class_names, pred_idx)
                    
                    if fig:
                        save_path = gradcam_dir / f"{img_path.stem}_gradcam.png"
                        fig.savefig(save_path, dpi=200, bbox_inches='tight')
                        plt.close(fig)
                        print(f"  Grad-CAM salvo em {save_path}")
    
    # Cria DataFrame a partir dos resultados
    df = pd.DataFrame(results)
    
    # Salva em CSV, se o arquivo de saída for especificado
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Resultados salvos em {output_file}")
    
    # Imprime resumo
    print("\n=== Resumo ===")
    print(f"Total de imagens processadas: {len(results)}")
    print("Predições por classe:")
    class_counts = df['predicted_class'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(results)*100:.1f}%)")
    
    return df

def main():
    """
    Interface de linha de comando para predição em lote.
    """
    parser = argparse.ArgumentParser(description='Predição em lote de imagens de raio-X de tórax em um diretório.')
    parser.add_argument('--dir', type=str, required=True,
                      help='Diretório contendo imagens de raio-X para predição')
    parser.add_argument('--output', type=str, default=None,
                      help='Caminho para o arquivo CSV de saída (opcional)')
    parser.add_argument('--ext', type=str, nargs='+', default=['.png', '.jpg', '.jpeg'],
                      help='Extensões de arquivos a serem processadas (padrão: .png .jpg .jpeg)')
    parser.add_argument('--save-gradcam', action='store_true',
                      help='Gera e salva visualizações Grad-CAM para cada imagem')
    
    args = parser.parse_args()
    
    # Executa predição em lote
    predict_batch(args.dir, args.output, args.ext, args.save_gradcam)

if __name__ == "__main__":
    main()
