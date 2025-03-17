# HMES-CNN: Handwritten Mathematical Expression Solver using CNN

A deep learning solution for recognizing and solving handwritten mathematical expressions using Convolutional Neural Networks.

## Overview

HMES-CNN is a complete pipeline for:
1. Detecting and segmenting handwritten mathematical symbols
2. Classifying symbols using a trained CNN model
3. Converting recognized symbols into LaTeX expressions
4. Solving or simplifying the mathematical expressions

## Features

- Symbol detection and segmentation using OpenCV
- CNN-based symbol classification
- Support for various mathematical symbols including:
  - Basic operators (+, -, ×, ÷)
  - Greek letters (π, θ)
  - Special symbols (∞, ≤, ≥)
  - Fractions and exponents
- LaTeX expression generation
- Mathematical expression solving using SymPy
- User-friendly web interface using Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aroshishp/Handwritten_Math_Expn_Solver.git
cd HMES-CNN
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Use the Jupyter notebook `hmes_cnn.ipynb` to train the CNN model:
```bash
jupyter notebook hmes_cnn.ipynb
```

### Running the Web Interface

Launch the Streamlit web application:
```bash
streamlit run streamlit_app.py
```

### Model Inference

Use `expn_infer.ipynb` for direct inference on images:
```bash
jupyter notebook expn_infer.ipynb
```

## Model Architecture

The CNN architecture consists of:
- 3 convolutional layers with ReLU activation and max pooling
- 3 fully connected layers with dropout
- Input size: 64x64 grayscale images
- Output: 12 symbol classes

## Project Structure

```
HMES-CNN/
│
├── Data Gen/
│   ├── hmes_cnn.ipynb        # Model training notebook
│   ├── expn_infer.ipynb      # Inference notebook
│   ├── streamlit_app.py      # Web interface
│   ├── requirements.txt      # Dependencies
│   └── README.md            # Documentation
│
└── models/
    └── final_math_symbol_cnn.pth  # Trained model weights
```

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- TensorFlow 2.7.0+
- OpenCV 4.5.3+
- Streamlit 1.8.0+
- Other dependencies listed in requirements.txt