# Predictive Maintenance Using XGBoost

This project implements a machine learning pipeline for Predictive Maintenance using the XGBoost classifier. The main objective is to predict equipment failures in industrial settings based on sensor readings and operational data, thereby minimizing downtime and reducing maintenance costs.

## 📖 Overview

Predictive maintenance is a technique that uses condition-monitoring tools and machine learning to track the performance of equipment during operation. This allows for timely maintenance to prevent unexpected failures. In this project, we use the **XGBoost** algorithm to classify whether a machine is likely to fail based on its sensor data and usage profile.

### Features include:

- `UDI`, `Product ID` (identifiers)
- `Type` (product type: L, M, H)
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Target` (binary: 1 = failure, 0 = no failure)
- `Failure Type` (textual label such as "No Failure", "Tool Wear Failure", etc.)

## 📁 Project Structure

```
Predictive-Maintenance-Using-XGBoost/
│
├── predictive_main.py           # Main script for preprocessing, training and evaluation
├── predictive_maintenance.csv   # Input dataset
├── confusion_mtx.png            # Confusion matrix (saved after training)
├── feature_imp.png              # Feature importance chart
└── README.md                    # Project documentation
```

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/vedantchavan004/Predictive-Maintenance-Using-XGBoost.git
cd Predictive-Maintenance-Using-XGBoost
```

2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## 🚀 Usage

Run the following command to execute the full pipeline:

```bash
python predictive_main.py
```

This script will:
- Load and clean the dataset
- Encode categorical variables
- Split the data into training/testing sets
- Train an XGBoost classifier
- Print classification metrics
- Save confusion matrix and feature importance plots

## 📈 Model Evaluation

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Feature Importance

## ✅ Results

### Confusion Matrix

![confusion_mtx](https://github.com/user-attachments/assets/a2578fbc-ea4e-40da-9f33-351a2784ef9c)

### Feature Importance

![feature_imp](https://github.com/user-attachments/assets/1a95b795-789b-4819-a827-044afba0a4c1)

These plots help visualize which features most influenced the model and how well it performed on unseen data.

