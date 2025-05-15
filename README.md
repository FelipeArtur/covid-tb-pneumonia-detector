# COVID-19, TB, and Pneumonia Detector

Deep learning model to detect COVID-19, Tuberculosis, Pneumonia, and Normal conditions from chest X-ray images.

---

## 🩺 **Project Overview**

This project uses a convolutional neural network based on MobileNetV2 to classify chest X-ray images into four categories:
- COVID-19
- Tuberculosis (TB)
- Pneumonia
- Normal (Healthy)

The model is trained on X-ray images and can be used to predict the condition from new images.

---

## 📁 **Project Structure**

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
└── README.md             # This file
```

---

## ⚠️ **Important Notes**

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

---

## 🛠️ **Installation**

### Automatic Installation (Recommended)

Use the installation scripts to create a virtual environment with `venv` and install dependencies automatically.

#### **Linux**

```bash
# Python 3.9 is required!
bash scripts/install.sh
```

#### **Windows**

```bat
REM Python 3.9 is required!
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

### Manual Installation (Alternative)

If you prefer, create the environment manually:

```bash
# Using venv
python3.9 -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate.bat # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Python 3.9 is required! TensorFlow 2.10 is not compatible with Python >=3.13.

---

## 📦 **Dataset**

Download X-ray datasets from:
- [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [Tuberculosis X-ray Images](https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

Organize the images according to the directory structure mentioned above.

---

## 🚀 **Usage**

### Training the Model

Basic training:
```bash
python src/model.py
```

Advanced training options:
```bash
# Use a different base model
python src/model.py --model resnet50v2

# Customize training parameters
python src/model.py --batch-size 16 --epochs 20 --img-size 299 299

# Disable fine-tuning or class weights
python src/model.py --no-fine-tuning --no-class-weights

# Control fine-tuning depth
python src/model.py --fine-tuning-layers 50
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
- Displays the image, probability graph, and textual result

#### Interpretation with Grad-CAM

```bash
python src/predict.py --image path/to/image.png --gradcam
```
- Displays Grad-CAM, overlay, and probability graph

### Batch Prediction

```bash
python src/predict_batch.py --dir path/to/images --output results.csv
```
- To save Grad-CAMs:
```bash
python src/predict_batch.py --dir path/to/images --output results.csv --save-gradcam
```

---

## 📊 **Performance**

The model achieves over 85% accuracy in classifying chest X-rays into the four categories.

Performance improvements:
- **Class weights** to handle the imbalance between classes
- **Fine-tuning** of base model layers for better feature extraction
- **Enhanced augmentation** for better generalization
- **Multiple backbone options** including MobileNetV2, ResNet50V2, EfficientNetB0, and InceptionV3
- **Learning rate scheduling** for improved convergence

Evaluation uses accuracy, precision, recall, F1-score, and confusion matrix, with results saved in the `results/` directory.

---

## 📝 **License**

MIT License - see the LICENSE file for details.

---

## 💡 **Troubleshooting**

- **Directory error:** Verify that the dataset structure is correct
- **Python version error:** Use Python 3.9
- **Memory issues:** Reduce batch size or use a GPU with more VRAM
- **Unexpected results:** Ensure that input images are similar to those in the training set

---