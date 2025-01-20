# Twitter Sentiment Analysis Using Multi-Layer Neural Network

This project implements a multi-layer neural network for sentiment analysis on Twitter data. The goal is to classify tweets into binary sentiment labels (e.g., positive or negative) based on their text features.

## Project Overview

- **Problem Statement**: Sentiment analysis is crucial for understanding public opinion in domains like marketing, politics, and product reviews. This project demonstrates the use of a custom neural network model to analyze Twitter data.
- **Model**: A custom multi-layer neural network with:
  - **Input Layer**: Size equal to the number of features in the dataset.
  - **Hidden Layers**: One or more layers with activation functions (ReLU, Sigmoid).
  - **Output Layer**: Produces binary sentiment labels (0 or 1).
- **Dataset**: Preprocessed Twitter dataset with text features and corresponding sentiment labels.

## Features

- **Custom Neural Network**:
  - Layers: Fully connected layers with activation functions.
  - Loss: Binary cross-entropy for classification.
  - Optimizer: Gradient descent-based optimizer.
- **Performance Metrics**:
  - Precision, Recall, F1-Score, and Accuracy on training and test sets.
- **Visualizations**:
  - Loss curves during training for model evaluation.

- **Custom Neural Network Training**:
  - The model is trained with preprocessed datasets, converting labels to binary `0` and `1`. A multi-layer neural network is implemented with the following components:
    - **Input Layer**: Accepts the input data size.
    - **Hidden Layers**: Includes configurable units using activation functions like ReLU and sigmoid.
    - **Output Layer**: Produces a binary classification result for each input.
  - **Optimization & Loss**: Binary cross-entropy loss is utilized with gradient-based optimization.

## Prerequisites

- Python 3.8+
- Required libraries:
  - NumPy
  - Scikit-learn
  - Matplotlib

## Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-repo-name/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```


Observations:
	•	Class imbalance in the dataset affects precision and recall.
	•	Adjusting the architecture or data preprocessing may yield better results.

File Structure
	•	project_rachamallu.ipynb: Jupyter notebook containing the code for data preprocessing, model training, and evaluation.
	•	data/: Directory for storing the dataset.
	•	results/: Directory for saving model outputs and performance visualizations.

Future Improvements
	•	Fine-tune the model architecture (e.g., number of layers, neurons, and activation functions).
	•	Improve data preprocessing to enhance feature extraction from tweets.
	•	Address class imbalance issues to improve precision and recall.

Contributors
	•	Yashashvini Rachamallu
Department of Computer Science and Engineering, Michigan State University
Email: rachama2@msu.edu
