#!/bin/bash

# Author: Felipe Artur Macedo Lima
# Date: 05-14-2025
# Description: This script installs the COVID-19, TB, and Pneumonia X-ray Analysis Tool on Linux.
# Usage: Run this script in the terminal to set up the environment and install dependencies.
# This script assumes that Python 3.9 and pip are already installed on the system.
# COVID-19, TB, and Pneumonia X-ray Analysis Tool - Linux Installation Script

echo "==================================================================="
echo "    Installing COVID-19, TB, and Pneumonia X-ray Analysis Tool      "
echo "==================================================================="

# Navigate to project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$PROJECT_ROOT"

echo -e "\n[1/4] Creating Python virtual environment..."
python3.9 -m venv venv || { echo "Failed to create virtual environment. Please make sure Python 3 is installed."; exit 1; }

echo -e "\n[2/4] Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate virtual environment."; exit 1; }

echo -e "\n[3/4] Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt || { echo "Failed to install required packages."; exit 1; }

echo -e "\n[4/4] Creating necessary directories..."
mkdir -p dataset/TRAIN
mkdir -p dataset/VAL
mkdir -p dataset/TEST
mkdir -p models
mkdir -p results

echo -e "\n==================================================================="
echo "                    Installation Complete!                           "
echo "==================================================================="
echo -e "\nTo use the application:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "2. Run the main application:"
echo "   python main.py"
echo -e "\nNOTE: Before training the model, you need to organize your dataset with the following structure:"
echo "  dataset/TRAIN/[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]"
echo "  dataset/VAL/[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]"
echo "  dataset/TEST/[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]"
echo -e "==================================================================="
```
