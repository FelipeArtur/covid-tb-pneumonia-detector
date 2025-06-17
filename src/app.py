from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from pathlib import Path
import logging
from werkzeug.utils import secure_filename
from predict import predict_image
from io import BytesIO
import base64
import tensorflow as tf
import numpy as np
from PIL import Image
from interpret import display_gradcam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path(__file__).parent.parent / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(filepath))
        
        # Make base prediction
        results = predict_image(str(filepath), show_plot=False)
        
        if results is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Check if Grad-CAM was requested
        if request.form.get('gradcam') == 'true':
            try:
                # Load the model
                model_path = Path(__file__).parent.parent / "models" / "best_model.h5"
                model = tf.keras.models.load_model(str(model_path))
                
                # Load and convert image to RGB if needed
                img = Image.open(filepath)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Prepare the image
                img_array = tf.keras.preprocessing.image.img_to_array(img.resize((224, 224))) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Generate Grad-CAM
                gradcam_results = display_gradcam(
                    model,
                    img,
                    img_array,
                    ["COVID", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
                )
                
                if gradcam_results:
                    # Convert overlay image to base64
                    buffered = BytesIO()
                    gradcam_results['overlay_img'].save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    results['gradcam_image'] = f'data:image/png;base64,{img_str}'
            except Exception as e:
                logger.error(f"Error generating Grad-CAM: {str(e)}")
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch', methods=['GET'])
def batch():
    return render_template('batch.html')

if __name__ == '__main__':
    app.run(debug=True)
