# Rice Disease Classification using Deep Learning

## Overview
This project implements a Convolutional Neural Network (CNN) to detect and classify rice diseases in the Philippines. Using transfer learning with VGG16 architecture, the model can identify 13 different rice diseases across fungal, bacterial, and viral categories.

## Problem Statement
Rice diseases pose a significant threat to Filipino agriculture and food security, leading to reduced yield, lower quality, and potential crop loss. This project aims to provide an accessible solution for early disease detection and intervention through deep learning-based classification.

## Dataset
The dataset is sourced from Omdena's Local Chapter project and includes:
- High-quality images of rice plants with various disease symptoms
- 224 x 224 pixel resolution
- Expert-validated classifications
- Balanced distribution across disease categories

### Disease Categories
1. **Fungal Diseases**
   - Rice Blast
   - Sheath Blight
   - Brown Spot
   - Narrow Brown Spot
   - Sheath Rot
   - Stem Rot
   - Bakanae
   - Rice False Smut

2. **Bacterial Diseases**
   - Bacterial Leaf Blight
   - Bacterial Leaf Streak

3. **Viral Diseases**
   - Tungro Virus
   - Ragged Stunt Virus
   - Grassy Stunt Virus

## Technical Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RCJamen/rice-classification.git
cd rice-classification
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # For Linux/Mac
# or
.venv\Scripts\activate  # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture
The model uses transfer learning with VGG16 as the base model:
1. Pre-trained VGG16 (weights frozen)
2. Flatten Layer
3. Dense Layer (512 units, ReLU activation)
4. Dropout Layer (0.5)
5. Output Layer (14 units, Softmax activation)

## Features
- Data augmentation for improved model generalization
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Model checkpointing for best weights
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- Visualization tools for dataset and predictions

## Usage
1. Prepare your dataset in the following structure:
```
dataset/
    disease_1/
        image1.jpg
        image2.jpg
        ...
    disease_2/
        image1.jpg
        image2.jpg
        ...
    ...
```

2. Run the Jupyter notebook:
```bash
jupyter notebook rice_classification_cnn.ipynb
```

## Model Performance
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

Actual performance metrics will vary based on your specific dataset and training conditions.

## Author
Ramel Cary B. Jamen (2019-2093)

## License
[Add your chosen license here]

## Acknowledgments
- Omdena's Local Chapter project for the dataset
- Agricultural experts who validated the disease classifications