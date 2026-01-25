"""
Aplicação web Flask para análise de raios-X.
"""
import os
import base64
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from src.core.config import CLASS_NAMES, MODEL_PATH, IMG_SIZE
from src.core.predict import predict_image
from src.core.interpret import display_gradcam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path(__file__).parent.parent.parent / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    """Verifica se a extensão do arquivo é permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Página inicial."""
    return render_template('index.html')


@app.route('/batch')
def batch():
    """Página de predição em lote."""
    return render_template('batch.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de predição."""
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo fornecido'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de arquivo inválido'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(filepath))
        
        results = predict_image(str(filepath), show_plot=False)
        
        if results is None:
            return jsonify({'error': 'Falha na predição'}), 500

        # Gerar Grad-CAM se solicitado
        if request.form.get('gradcam') == 'true':
            gradcam_image = _generate_gradcam(filepath)
            if gradcam_image:
                results['gradcam_image'] = gradcam_image
        
        os.remove(filepath)
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Erro durante predição: {str(e)}")
        return jsonify({'error': 'Erro interno do servidor'}), 500


def _generate_gradcam(filepath):
    """Gera visualização Grad-CAM para uma imagem."""
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        
        img = Image.open(filepath)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = tf.keras.preprocessing.image.img_to_array(img.resize(IMG_SIZE)) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        gradcam_results = display_gradcam(model, img, img_array, CLASS_NAMES)
        
        if gradcam_results:
            buffered = BytesIO()
            gradcam_results['overlay_img'].save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f'data:image/png;base64,{img_str}'
            
    except Exception as e:
        logger.error(f"Erro ao gerar Grad-CAM: {str(e)}")
    
    return None


if __name__ == '__main__':
    app.run(debug=True)
