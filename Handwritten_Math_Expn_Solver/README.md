# Handwritten Mathematical Expression Solver using CNN

A deep learning solution for recognizing and solving handwritten mathematical expressions using Convolutional Neural Networks.

## Overview

Handwritten Mathematical Expression Solver is a complete pipeline for:
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

### Solver Features

- Linear Equations
- Roots of polynomials
- Exponential Equations
- Simplification of plain expressions
- Rational Equations
- Multivariate Equations
- Check if equality is true/false

### UI Features

- Allow user to upload images
- Get the predicted LaTeX expression, that can be copied to the clipboard directly
- The predicted LaTeX is rendered on the screen for easy checking
- The user has the option to manually correct the expression, in case of mis-identification
- An option to solve/simplify the equation/expression

## Installation

1. Clone the repository:
```bash
git clone https://github.com//IITH-Epoch/Projects_2024-25.git
cd Projects_2024-25/Handwritten_Math_Expn_Solver
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
- Output: 49 symbol classes

## Demo Video
![Demo](https://github.com/IITH-Epoch/Projects_2024-25/blob/main/Handwritten_Math_Expn_Solver/HMES%20Demo%203.gif)

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- TensorFlow 2.7.0+
- OpenCV 4.5.3+
- Streamlit 1.8.0+
- Other dependencies listed in requirements.txt
