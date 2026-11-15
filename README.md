# ML Sentiment Analysis

This project builds and compares recurrent neural network models for multiclass sentiment classification using a text dataset. The notebook loads a sentiment dataset, preprocesses the text, tunes model hyperparameters, evaluates several architectures, and visualizes the results.

## Overview

The repository contains a single notebook, `sentiment_analysis.ipynb`, which explores:

- text preprocessing and encoding
- tokenization with padding
- hyperparameter tuning using Keras Tuner
- comparison of RNN, LSTM, and GRU architectures
- evaluation using accuracy, precision, recall, and macro F1
- confusion matrices and training/validation plots

## Project Goal

The goal is to classify text into sentiment categories such as:

- Negative
- Neutral
- Positive

The model is trained on a publicly available multiclass sentiment dataset and evaluated on a held-out test set.

## Dataset

The notebook uses the Hugging Face dataset:

- `Sp1786/multiclass-sentiment-analysis-dataset`

It includes text samples and corresponding labels, with three sentiment classes.

## Models Compared

The project experiments with:

- Simple RNN
- LSTM
- GRU

Each architecture is tuned with Keras Tuner using a random search strategy, with the objective set to maximize validation macro F1.

## Technologies Used

- Python
- TensorFlow / Keras
- Keras Tuner
- Hugging Face `datasets`
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Repository Structure

```text
ML/
├── README.md
├── sentiment_analysis.ipynb
└── (generated tuning outputs and model artifacts when run locally)
```

## Setup

### Option 1: Google Colab

This notebook was originally developed in a Colab environment, so it runs well there with minimal setup.

### Option 2: Local Python Environment

1. Clone the repository:

```bash
git clone https://github.com/GAJIN-S/ML.git
cd ML
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -q datasets tensorflow keras-tuner scikit-learn pandas matplotlib seaborn
```

4. Open `sentiment_analysis.ipynb` in Jupyter Notebook or VS Code, then run the cells sequentially.

## Usage

Open the notebook and execute the cells in order:

1. install dependencies
2. import libraries
3. load the dataset
4. preprocess text and labels
5. tokenize and pad sequences
6. tune models with Keras Tuner
7. train final models
8. compare metrics and visualize results

## Example Results

From the notebook run, the best-performing architecture on the test set was:

- GRU: Macro F1 = 0.6844
- LSTM: Macro F1 = 0.6786
- RNN: Macro F1 = 0.6717

These numbers reflect the model's multiclass sentiment classification performance on the selected dataset.

## Notes

- The notebook uses a fixed random seed for reproducibility.
- Early stopping and learning-rate reduction are used to improve generalization.
- Hyperparameter tuning is designed to optimize validation macro F1, which is especially useful in class-imbalanced classification tasks.

## Future Improvements

Possible next steps for this project include:

- trying transformer-based models such as BERT
- tuning additional hyperparameters
- evaluating on a larger balanced dataset
- adding a simple inference script for real-time sentiment prediction
- exporting the trained model for deployment

## Contact

This repository appears to be a personal learning project. If you want to extend or improve it, feel free to fork the project and submit your changes.
