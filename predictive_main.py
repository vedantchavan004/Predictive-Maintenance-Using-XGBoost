# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('predictive_maintenance.csv')

# Explore the dataset
print("Dataset Info:")
print(data.info())
print("\nFirst 5 Rows of Dataset:")
print(data.head())

# Drop irrelevant columns
data.drop(['UDI', 'Product ID'], axis=1, inplace=True)  # Remove unnecessary columns

# Encode categorical features
data['Type'] = data['Type'].astype('category').cat.codes  # Encode 'Type'
data['Failure Type'] = data['Failure Type'].astype('category').cat.codes  # Encode 'Failure Type'

# Define features and target
X = data.drop(['Target'], axis=1)  # Features
y = data['Target']  # Target variable

# Clean column names for compatibility with XGBoost
X.columns = X.columns.str.replace(r"[^\w\s]", "", regex=True)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Air temperature K', 'Process temperature K', 'Rotational speed rpm', 
                      'Torque Nm', 'Tool wear min']

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Initialize and train the XGBoost Classifier
xgb_model = XGBClassifier(
    objective='binary:logistic', 
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    random_state=42,
    use_label_encoder=False,  # Prevent warnings
    eval_metric='logloss'  # Evaluation metric
)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate model performance
print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_model.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
