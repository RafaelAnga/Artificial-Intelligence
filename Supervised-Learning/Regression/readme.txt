
################################################################################################################
# K-Nearest Neighbors (KNN) Social Network Ad Prediction

## Project Overview
This project implements a K-Nearest Neighbors (KNN) classifier to predict whether users will purchase based on their age and estimated salary from social network ad data. The model helps identify potential customers for targeted advertising.

## Technical Stack
- Python 3.x
- Libraries:
  - scikit-learn: For KNN implementation and model evaluation
  - Pandas: Data manipulation and analysis
  - NumPy: Numerical computations
  - Matplotlib: Visualization
  - Google Colab: Development environment

## Dataset
The project uses 'Social_Network_Ads.csv' containing:
- Age
- Estimated Salary
- Purchase Decision (0 = No, 1 = Yes)

## Features
1. Data Preprocessing
   - Feature scaling using StandardScaler
   - Train-test split (75%-25%)
   - Data normalization

2. Model Implementation
   - KNN Classifier with parameters:
     - n_neighbors = 5
     - metric = 'minkowski'
     - p = 2 (Euclidean distance)

3. Visualization
   - Decision boundary plotting
   - Training set visualization
   - Test set visualization
   - Confusion Matrix display

## Model Performance
- Confusion Matrix analysis
- Accuracy score calculation
- Visual representation of predictions
- Decision boundary visualization
