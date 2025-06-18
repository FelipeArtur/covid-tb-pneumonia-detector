# X-Ray Analysis Tool

## Overview
This web application analyzes chest X-ray images to detect COVID-19, Tuberculosis (TB), and Pneumonia using deep learning. Built with Flask and TensorFlow, it provides an intuitive interface for medical professionals to upload and analyze X-ray images.

## Features
- **Web Interface**: User-friendly upload and analysis of chest X-rays
- **Real-time Analysis**: Instant results with confidence scores
- **Grad-CAM Visualization**: Heat map visualization of regions of interest
- **Batch Processing**: Support for analyzing multiple images
- **Responsive Design**: Works on desktop and mobile devices

## Setup

### Requirements
- Python 3.9
- pip
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/covid-tb-pneumonia-detector.git
cd covid-tb-pneumonia-detector
```

2. Run the installation script:
```bash
./scripts/install.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

4. Start the application:
```bash
python src/app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure
```
covid-tb-pneumonia-detector/
├── models/               # Trained model files
├── src/                 
│   ├── static/          # CSS, JS, and images
│   ├── templates/       # HTML templates
│   ├── app.py           # Flask application
│   ├── predict.py       # Prediction logic
│   └── interpret.py     # Grad-CAM visualization
├── uploads/             # Temporary upload directory
├── scripts/            
│   └── install.sh       # Installation script
└── requirements.txt     # Dependencies
```

## Usage
1. Access the web interface at `http://localhost:5000`
2. Upload an X-ray image using drag-and-drop or file selection
3. Toggle Grad-CAM visualization if desired
4. View the analysis results and confidence scores
5. For batch processing, use the "Batch Processing" tab

## Technical Details
- Backend: Flask
- Deep Learning: TensorFlow 2.10
- Frontend: HTML5, CSS3, JavaScript
- Model: MobileNetV2 architecture with transfer learning


## Important Considerations
1. **Dataset Preparation**
   - Follow the directory structure exactly as shown above
   - Subdirectory names must be: `COVID`, `NORMAL`, `PNEUMONIA`, `TUBERCULOSIS`
   - Images should be in standard formats (JPG, PNG, BMP)

2. **Validation**
   - The code validates the existence of essential directories and files before running
   - Clear error messages are displayed if anything is missing

3. **Training**
   - Requires a GPU with at least 4GB of VRAM for adequate performance
   - Training time depends on the dataset size

4. **Prediction**
   - The model only works correctly with images similar to those in the training set
   - Images should be properly oriented and of good quality

## Usage Instructions

### Training the Model

Basic training:
```bash
python src/model.py
```

Available base models:
- mobilenetv2 (default)
- resnet50v2
- efficientnetb0
- inceptionv3

The model will be saved as `models/best_model.h5` and metrics/graphs will be saved in `results/`.


## Key Files

| File | Description |
|------|-------------|
| [model.py](/home/felipe/Documents/Projetos/covid-tb-pneumonia-detector/src/model.py) | Defines the neural network architecture, training process, and evaluation metrics. Supports multiple CNN backbones. |
| [predict.py](/home/felipe/Documents/Projetos/covid-tb-pneumonia-detector/src/predict.py) | Handles single image prediction with visualization options. Contains the `load_and_preprocess_image()` and `predict_image()` functions. |
| [predict_batch.py](/home/felipe/Documents/Projetos/covid-tb-pneumonia-detector/src/predict_batch.py) | Processes multiple images in a directory, generates CSV reports and optionally creates Grad-CAM visualizations. |
| [interpret.py](/home/felipe/Documents/Projetos/covid-tb-pneumonia-detector/src/interpret.py) | Implements Grad-CAM visualization techniques to highlight regions of interest in X-ray images that influenced predictions. |
| [install.sh](/home/felipe/Documents/Projetos/covid-tb-pneumonia-detector/scripts/install.sh) | Linux installation script for setting up the required Python environment and dependencies. |
| [install.bat](/home/felipe/Documents/Projetos/covid-tb-pneumonia-detector/scripts/install.bat) | Windows installation script for setting up the required Python environment and dependencies. |
| [requirements.txt](/home/felipe/Documents/Projetos/covid-tb-pneumonia-detector/requirements.txt) | Lists all Python dependencies required by the project, including specific versions for compatibility. |

## Performance

The model achieves over 85% accuracy in classifying chest X-rays into the four categories.

Performance improvements:
- **Class weights** to handle the imbalance between classes
- **Fine-tuning** of base model layers for better feature extraction
- **Enhanced augmentation** for better generalization
- **Multiple backbone options** including MobileNetV2, ResNet50V2, EfficientNetB0, and InceptionV3
- **Learning rate scheduling** for improved convergence

Evaluation uses accuracy, precision, recall, F1-score, and confusion matrix, with results saved in the `results/` directory.

## Troubleshooting

- **Directory error:** Verify that the dataset structure is correct
- **Python version error:** Use Python 3.9
- **Memory issues:** Reduce batch size or use a GPU with more VRAM
- **Unexpected results:** Ensure that input images are similar to those in the training set

## Team

This project is maintained by:

* **Lead Developer**: Felipe Lima and Rafael Miguez
* **Contributors**: Danilo Scheltes, Felipe Lima and Rafael Miguez
