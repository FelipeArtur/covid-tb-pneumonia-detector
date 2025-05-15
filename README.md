# COVID-19, Tuberculosis, and Pneumonia Detector

## Overview
This repository contains a deep learning model that classifies chest X-ray images into four categories: COVID-19, Tuberculosis (TB), Pneumonia, and Normal (healthy). The project was developed using a convolutional neural network architecture based on transfer learning techniques.

The **COVID-19, TB, and Pneumonia Detector** was created to assist medical professionals in the initial screening of chest X-ray images, potentially identifying patterns associated with respiratory conditions. By leveraging deep learning, the system can help prioritize cases that may require urgent medical attention, especially in resource-constrained settings.

## Features
- **Multi-class Classification**: Differentiates between four distinct conditions (COVID-19, TB, Pneumonia, and Normal) with high accuracy.
- **Transfer Learning Model**: Utilizes pre-trained CNN architectures (MobileNetV2, ResNet50V2, EfficientNetB0, or InceptionV3) fine-tuned on X-ray images.
- **Grad-CAM Visualization**: Provides heatmap visualization to highlight regions of interest that influenced the model's decision.
- **Batch Processing**: Supports processing multiple images at once with comprehensive reporting.
- **Flexible Deployment**: Works with various image formats (JPG, PNG, BMP) and provides clear prediction results.
- **Performance Metrics**: Generates detailed performance reports including accuracy, precision, recall, and F1-score.

## Setup Process
Before running the application, you need to set up the environment and prepare the dataset.

### Installation Process

#### Automatic Installation (Recommended)

Use the installation scripts to create a virtual environment with `venv` and install dependencies automatically.

##### Linux
```bash
bash scripts/install.sh
```

##### Windows
```bat
scripts\install.bat
```

After installation, activate the virtual environment:

- **Linux:**  
  ```bash
  source venv/bin/activate
  ```
- **Windows:**  
  ```bat
  venv\Scripts\activate.bat
  ```

#### Manual Installation (Alternative)

If you prefer, create the environment manually:

```bash
python3.9 -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate.bat # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Python 3.9 is required! TensorFlow 2.10 is not compatible with newer versions of Python.

### Dataset Preparation
1. Download X-ray datasets from:
   - [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
   - [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
   - [Tuberculosis X-ray Images](https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

2. Organize the images according to the directory structure specified in the Project Structure section.

## Project Structure
The repository is organized with a clear structure to separate code, data, and outputs.

```
covid-tb-pneumonia-detector/
├── dataset/              # Dataset (you need to create this)
│   ├── TRAIN/
│   │   ├── COVID/
│   │   ├── NORMAL/
│   │   ├── PNEUMONIA/
│   │   └── TUBERCULOSIS/
│   ├── VAL/
│   │   ├── COVID/
│   │   ├── NORMAL/
│   │   ├── PNEUMONIA/
│   │   └── TUBERCULOSIS/
│   └── TEST/
│       ├── COVID/
│       ├── NORMAL/
│       ├── PNEUMONIA/
│       └── TUBERCULOSIS/
├── models/               # Trained models
├── results/              # Results and metrics
├── src/                  # Source code
│   ├── model.py          # Model definition and training
│   ├── predict.py        # Single image prediction
│   ├── predict_batch.py  # Batch prediction
│   └── interpret.py      # Model interpretation
├── scripts/              # Installation scripts
│   ├── install.sh        # Linux installation
│   └── install.bat       # Windows installation
├── requirements.txt      # Dependencies
└── README.md             
```

### Important Considerations
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

### Individual Prediction

```bash
python src/predict.py --image path/to/image.png
```

This will:
- Display the X-ray image
- Show prediction probabilities for all classes
- Highlight the predicted class with confidence score
- Print detailed results to the console

#### Visualization with Grad-CAM

```bash
python src/predict.py --image path/to/image.png --gradcam
```

This enhanced visualization includes:
- Original X-ray image
- Grad-CAM heatmap showing important regions for the prediction
- Overlay visualization with highlighted areas
- Prediction probabilities for all classes

### Batch Prediction

```bash
python src/predict_batch.py --dir path/to/images --output results.csv
```

To save Grad-CAM visualizations for each image:
```bash
python src/predict_batch.py --dir path/to/images --output results.csv --save-gradcam
```

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
