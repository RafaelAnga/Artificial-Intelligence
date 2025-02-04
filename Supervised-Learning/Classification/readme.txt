Bank Customer Churn Prediction using Artificial Neural Network

**Project Overview**
This project implements an Artificial Neural Network (ANN) to predict customer churn in a banking context. Customer churn refers to customers leaving the bank, and this model helps identify such customers based on their demographic, financial, and behavioral data. By predicting churn, banks can take proactive measures to retain valuable customers and improve customer satisfaction.

Technical Stack
- Python 3.x
- Libraries:
- TensorFlow: For building and training the ANN
- Pandas: Data manipulation and analysis
- NumPy: Numerical operations
- Scikit-learn: Data preprocessing, feature scaling, and evaluation
- Matplotlib & Seaborn: Visualization
- Google Colab: Development environment

## Dataset
10,000 customer records with the following features:
- Demographics (Geography, Gender, Age)
- Banking relationship (Credit Score, Balance, Products)
- Account details (Tenure, Credit Card status)
- Customer behavior (IsActiveMember)
- Economic factors (EstimatedSalary)
- Target variable: Binary (0 = customer stays, 1 = customer churns)

## Features
1. Data Preprocessing
   - Removal of non-predictive columns (CustomerId, Surname)
   - Categorical encoding for Geography and Gender
   - Feature scaling and preparation
   - Splitting the dataset into 80% training and 20% testing sets

2.ANN Architecture
  - Input layer: Accepts all features
  - Two hidden layers:
    - Each with 6 neurons and ReLU activation function
  - Output layer:
    - 1 neuron with Sigmoid activation for binary classification

3. Model Training
    - Optimizer: Adam (adaptive learning rate optimization)
    - Loss function: Binary Cross-Entropy (for binary classification)
    - Metrics: Accuracy
    - Training: 100 epochs with a batch size of 32

4.Prediction and Evaluation
    - Predicts churn probability for individual customers
    - Evaluates model performance on the test set
    - Confusion Matrix and Accuracy Score for performance metrics
    - Visualizes confusion matrix using Seaborn heatmap

## Key Metrics
- Confusion Matrix: Displays true positives, true negatives, false positives, and false negatives
- Accuracy Score: Quantifies the overall prediction success of the mode

### Business Applications
- Customer Retention: Identifies customers likely to churn, enabling targeted retention strategies
- Resource Allocation: Helps allocate resources to high-risk customers
- Data-Driven Decision Making: Supports marketing and customer service teams with actionable insights
- Revenue Optimization: Reduces churn rates, leading to increased customer lifetime value

#############################################################################################################################################################################################################################

# Bank Customer Churn Prediction using XGBoost

## Project Overview
This project implements a machine learning solution using XGBoost to predict customer churn in a banking context. The model analyzes various customer attributes to predict whether a customer is likely to leave the bank, enabling proactive customer retention strategies.

## Technical Stack
- Python 3.x
- Libraries:
  - XGBoost (v2.1.2): Core classification algorithm
  - scikit-learn (v1.5.2): For model evaluation and data processing
  - Pandas: Data manipulation and analysis
  - NumPy: Numerical operations
  - Matplotlib: Visualization
  - Google Colab: Development environment

## Dataset
The project uses 'churn_modelling.csv' containing customer information:
- Demographics (Geography, Gender, Age)
- Banking relationship (Credit Score, Balance, Products)
- Account details (Tenure, Credit Card status)
- Customer behavior (IsActiveMember)
- Economic factors (EstimatedSalary)

## Features
1. Data Preprocessing
   - Removal of non-predictive columns (CustomerId, Surname)
   - Categorical encoding for Geography and Gender
   - Feature scaling and preparation

2. Model Implementation
   - XGBoost Classifier with optimized parameters:
     - max_depth = 3
     - learning_rate = 0.2
     - n_estimators = 100
     - subsample = 1.0

3. Model Evaluation
   - Confusion Matrix visualization
   - Accuracy metrics
   - 10-fold Cross-validation
   - Hyperparameter optimization using GridSearchCV

## Performance Metrics
- High accuracy on test data
- Robust cross-validation results
- Detailed confusion matrix analysis
- Optimized hyperparameters through grid search

#############################################################################################################################################################################################################################

# Breast Cancer Detection using XGBoost

## Project Overview
This machine learning project implements a breast cancer detection system using the XGBoost algorithm to classify breast cancer cases as benign or malignant. The model achieves high accuracy in predicting cancer diagnoses based on cellular characteristics.

## Technical Stack
- Python 3.x
- Key Libraries:
  - XGBoost (v2.1.2): Core classification algorithm
  - scikit-learn (v1.5.1): For model evaluation and data splitting
  - Pandas: Data manipulation and analysis
  - NumPy: Numerical computations
  - Matplotlib: Visualization
  
## Dataset
The project uses the Breast Cancer Wisconsin dataset containing cellular features:
- Features include measurements of cell characteristics
- Binary classification: Benign (0) or Malignant (1)
- Dataset is split 80/20 for training and testing

## Features
1. Data Preprocessing
   - Data loading and exploration
   - Binary label conversion
   - Train-test splitting

2. Model Implementation
   - XGBoost Classifier with logloss evaluation metric
   - Hyperparameter optimization
   - Model training and prediction

3. Performance Evaluation
   - Confusion Matrix visualization
   - Accuracy metrics
   - K-Fold Cross-validation (10 folds)

## Model Performance
- High accuracy on test data
- Robust performance verified through cross-validation
- Mean accuracy and standard deviation calculations
- Confusion matrix visualization for detailed performance analysis

## Usage
1. Install required dependencies:
pip install xgboost==2.1.2
pip install scikit-learn==1.5.1

2. Load and preprocess the breast cancer dataset
3. Train the XGBoost model
4. Evaluate results using provided metrics

## Key Results
- Reliable cancer detection through machine learning
- Cross-validated performance metrics
- Visualization of model predictions
- Robust evaluation through multiple validation techniques

## Applications
- Medical diagnosis support
- Cancer screening assistance
- Research and development in medical AI
- Educational purposes in medical data analysis

## Future Improvements
- Feature importance analysis
- Hyperparameter tuning optimization
- Integration with medical imaging data
- Development of user interface for medical professionals

#############################################################################################################################################################################################################################

# Kernel SVM Social Network Ad Prediction

## Project Overview
This project implements a Kernel Support Vector Machine (SVM) classifier to predict customer purchases based on social network advertising data, using age and estimated salary as features.

## Technical Stack
- Python 3.x
- Libraries:
  - scikit-learn: For SVM implementation
  - Pandas: Data manipulation
  - NumPy: Numerical operations
  - Matplotlib: Visualization
  - Google Colab: Development environment

## Dataset
Uses 'Social_Network_Ads.csv' containing:
- Age
- Estimated Salary
- Purchase Decision (0/1)

## Features
1. Data Preprocessing
   - Feature scaling
   - Train-test split (75-25)

2. Model Implementation
   - Kernel SVM with RBF kernel
   - Random state for reproducibility

3. Visualization
   - Decision boundary plotting
   - Training/Test set visualization
   - Confusion Matrix


############################################################################################################################################################################################################################
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
