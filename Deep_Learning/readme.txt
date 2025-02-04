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
