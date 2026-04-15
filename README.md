# Vehicle Insurance Fraud Detection

An end-to-end **Machine Learning and Deep Learning** project that detects fraudulent vehicle insurance claims using claim, policy, and vehicle-related data. This system helps identify suspicious claims early, supporting faster investigation and better fraud prevention.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Challenges Addressed](#key-challenges-addressed)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Author](#author)
- [License](#license)

## Overview

Insurance fraud is a major challenge for the insurance industry, leading to significant financial losses and inefficient claim processing. This project focuses on building an intelligent fraud detection system that classifies vehicle insurance claims as **fraudulent** or **genuine** using historical insurance data.

The project combines traditional machine learning approaches with an **LSTM-based deep learning model** to improve prediction performance and support better fraud analysis.

## Problem Statement

Vehicle insurance providers receive a large number of claims, and manually identifying fraudulent ones is time-consuming and error-prone. The goal of this project is to automate fraud detection by building predictive models that can learn hidden patterns from claim data and flag suspicious records effectively.

## Objectives

- Build a system to detect fraudulent vehicle insurance claims
- Preprocess and transform raw insurance data into usable features
- Handle data imbalance between fraudulent and non-fraudulent claims
- Train and compare multiple machine learning models
- Implement an LSTM model for deep learning-based prediction
- Evaluate model performance using standard classification metrics

## Features

- Comprehensive data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Handling of imbalanced data
- Multiple classification models
- Deep learning with LSTM
- Model evaluation and comparison
- Fraud prediction pipeline

## Tech Stack

**Languages & Libraries**  
Python, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow, Keras

**Deep Learning**  
LSTM Neural Network

**Visualization**  
Matplotlib, Seaborn

**Development Environment**  
Jupyter Notebook, GitHub

## Project Workflow

```text
Data Collection
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Handling Class Imbalance
   ↓
Model Training
   ↓
LSTM Model Building
   ↓
Model Evaluation
   ↓
Fraud Prediction
```

## Models Used

### Machine Learning Models
- XGBoost Classifier

### Deep Learning Model
- LSTM Neural Network

## Evaluation Metrics

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

## Key Challenges Addressed

- Dealing with imbalanced fraud detection data
- Cleaning and encoding complex insurance claim features
- Extracting meaningful patterns from structured claim records
- Comparing machine learning and deep learning approaches
- Reducing false positives while improving fraud detection capability

## Project Structure

```bash
vehicle_insurance_fraud_detection/
│
├── data/                # Dataset files
├── notebooks/           # Jupyter notebooks for analysis and modeling
├── src/                 # Source code scripts
├── models/              # Saved trained models
├── outputs/             # Graphs, reports, predictions
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/hineni26/vehicle_insurance_fraud_detection.git
cd vehicle_insurance_fraud_detection
pip install -r requirements.txt
```

## Usage

You can run the project through Jupyter Notebook or Python scripts.

### Launch Jupyter Notebook

```bash
jupyter notebook
```

### Typical Steps
- Load the dataset
- Preprocess the data
- Perform EDA
- Train machine learning models
- Train the LSTM model
- Evaluate all models
- Predict fraudulent claims

## Results

This project demonstrates how machine learning and deep learning can be used to identify suspicious vehicle insurance claims efficiently. By combining preprocessing, feature engineering, and model comparison, the system supports more accurate fraud detection and improved claim screening.

## Future Enhancements

- Deploy the model as a web application
- Add real-time fraud prediction support
- Improve LSTM architecture tuning
- Apply advanced imbalance handling techniques such as SMOTE
- Integrate explainable AI methods for model interpretability
- Build an end-to-end MLOps pipeline

## Author

**Ahan Mondal**

## License

This project is intended for educational and research purposes.