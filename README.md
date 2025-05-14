# COVID-19, TB, and Pneumonia Detector

Deep learning model to detect COVID-19, Tuberculosis, Pneumonia, and Normal conditions from chest X-ray images.

## Project Overview

This project uses a MobileNetV2-based convolutional neural network to classify chest X-ray images into four categories:
- COVID-19
- Tuberculosis (TB)
- Pneumonia
- Normal

The model is trained on X-ray images and can be used to predict the condition of new, unseen X-ray images.

## Project Structure

```
covid-tb-pneumonia-detector/
├── dataset/              # Dataset directory (you must create this)
│   ├── TRAIN/            # Training images
│   │   ├── COVID/        # COVID-19 X-ray images
│   │   ├── NORMAL/       # Normal X-ray images
│   │   ├── PNEUMONIA/    # Pneumonia X-ray images
│   │   └── TUBERCULOSIS/ # Tuberculosis X-ray images
│   ├── VAL/              # Validation images (same structure as TRAIN)
│   └── TEST/             # Test images (same structure as TRAIN)
├── models/               # Directory to store trained models
├── results/              # Directory to store evaluation results
├── src/                  # Source code
│   ├── model.py          # Model definition and training
│   ├── predict.py        # Prediction functionality
│   └── interpret.py      # Model interpretability (Grad-CAM visualization)
├── scripts/              # User scripts
│   └── predict_batch.py  # Script for batch image prediction
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Important Notes for Project Setup

1. **Dataset Preparation**: This is the most critical step!
   - The dataset must follow the exact directory structure shown above
   - All subdirectories must have identical class names (COVID, NORMAL, PNEUMONIA, TUBERCULOSIS)
   - Images should be in standard formats (JPG, PNG)

2. **Model Training**:
   - Training requires sufficient GPU resources - at least 4GB VRAM is recommended
   - Initial training may take several hours depending on dataset size

3. **Predictions**:
   - The model will only work with chest X-ray images similar to the training data
   - Images must be properly oriented and of sufficient quality

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/covid-tb-pneumonia-detector.git
cd covid-tb-pneumonia-detector
```

2. **IMPORTANT: Python 3.9 Required**
   TensorFlow is only compatible with Python up to version 3.9 and will not work with Python 3.13.3.

   Option 1: If you have Python 3.9 installed:
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Option 2: Using Conda (recommended):
   ```bash
   conda create -n covid-detector python=3.9
   conda activate covid-detector
   pip install -r requirements.txt
   ```

   Option 3: Using Docker:
   ```bash
   # A Dockerfile is provided in the repository
   docker build -t covid-detector .
   docker run -it covid-detector
   ```

3. Run tests to ensure the setup is correct:
```bash
pytest
```

4. Download the dataset (instructions below) and place it in the `dataset` directory.

## Python Version

**This project requires Python 3.9 specifically.**

TensorFlow is not compatible with Python 3.13.3. You must use Python 3.9 to run this project.

## Dataset

You can download chest X-ray datasets from:
- [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [Tuberculosis X-ray Images](https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

After downloading, organize the images into the required directory structure described above.

## Usage

### Training the Model

To train the model:

```bash
python src/model.py
```

The training process will:
1. Load images from the dataset directory
2. Train a MobileNetV2-based model
3. Save the best model to `models/best_model.h5`
4. Generate performance metrics in the `results` directory

### Making Single Predictions

To predict a single image:

```bash
python src/predict.py --image path/to/your/image.png
```

This will display:
- The original X-ray image
- A bar chart showing prediction probabilities
- Text output with detailed prediction results

#### Model Interpretability with Grad-CAM

For better understanding of what the model is focusing on:

```bash
python src/predict.py --image path/to/your/image.png --gradcam
```

This will display:
- The original X-ray image
- A Grad-CAM heatmap showing important regions for the prediction
- An overlay of the heatmap on the original image
- A bar chart showing prediction probabilities

### Batch Predictions

To predict multiple images in a directory:

```bash
python scripts/predict_batch.py --dir path/to/images --output results.csv
```

To also generate Grad-CAM visualizations for all images:

```bash
python scripts/predict_batch.py --dir path/to/images --output results.csv --save-gradcam
```

This will:
- Process all images in the specified directory
- Output results to the console
- Save detailed results to a CSV file if `--output` is specified
- Save Grad-CAM visualizations to a subdirectory if `--save-gradcam` is specified

## Performance

The model is evaluated using accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visualize the performance.

Training results and model performance metrics will be saved in the `results` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.