# ML Sentiment Analysis

A machine learning project that compares **RNN, LSTM, and GRU** models for multiclass sentiment analysis.

The models classify text into three categories:
- Negative
- Neutral
- Positive

## Dataset

[Multiclass Sentiment Analysis – Hugging Face](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset)

## Technologies Used

- Python
- TensorFlow / Keras
- Keras Tuner
- Hugging Face Datasets
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## Models Compared

- Simple RNN
- LSTM
- GRU

Hyperparameter tuning is performed using Keras Tuner with Random Search.

## Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| RNN | 66.82% | 67.17% |
| LSTM | 67.82% | 67.86% |
| **GRU** | **68.26%** | **68.44%** |

**Best Model:** GRU

entiment classification.
- Build a sentiment prediction application.
