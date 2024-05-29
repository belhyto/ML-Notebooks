import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, 
accuracy_score, f1_score


# Load the dataset
data = pd.read_csv("/kaggle/input/heart-disease-prediction-using-logisticregression/framingham.csv")

# Splitting the data into features and target variable
X = data.drop('TenYearCHD', axis=1) # Features
y = data['TenYearCHD'] # Target variable

# Create a pipeline with an imputer transformer and logistic regression model
pipeline = make_pipeline(SimpleImputer(strategy='mean'), LogisticRegression(max_iter=1000))
# Fitting the pipeline
pipeline.fit(X, y)


# Predicting on the entire dataset
y_pred = pipeline.predict(X)

# Calculating precision, recall, accuracy, F1 score, and support
precision = precision_score(y, y_pred)


recall = recall_score(y, y_pred)
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
support = np.sum(y == 1) # Total number of positive instances
# Printing classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))
# Printing confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))
# Printing evaluation metrics
print("\nPrecision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Support:", support)
