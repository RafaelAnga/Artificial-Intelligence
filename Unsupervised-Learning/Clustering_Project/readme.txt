# Customer Segmentation Analysis using K-Means Clustering

## Project Overview
This project implements customer segmentation analysis using K-Means clustering algorithm to identify distinct customer groups based on their annual income and spending patterns. The analysis helps businesses understand their customer base and develop targeted marketing strategies.

## Technical Stack
- Python 3.x
- Libraries:
  - NumPy: For numerical computations
  - Pandas: For data manipulation and analysis
  - Matplotlib: For data visualization
  - Seaborn: For enhanced visualizations
  - Scikit-learn: For implementing K-Means clustering
  - Yellowbrick: For elbow method visualization

## Dataset
The analysis uses 'Mall_Customers.csv' which contains customer information including:
- Annual Income (k$)
- Spending Score (1-100)

## Features
1. Data Preprocessing and Exploration
   - Dataset loading and initial exploration
   - Feature selection for clustering

2. Optimal Cluster Selection
   - Implementation of elbow method
   - Visual determination of optimal number of clusters

3. K-Means Clustering Implementation
   - Model training with optimal cluster number (k=5)
   - Customer segmentation based on income and spending patterns

4. Visualization
   - Cluster visualization with distinct colors
   - Centroid marking for each cluster
   - Pairplot analysis of relationships in data

## Analysis Results
The analysis identifies 5 distinct customer segments:
1. Low income (20K-40K), low spending score (0-40)
2. Low income (20K-40K), high spending score (60-100)
3. Average income (40K-75K), average spending score (40-60)
4. High income (75K-140K), low spending score (0-40)
5. High income (75K-140K), high spending score (60-100)

## Business Applications
- Targeted marketing strategies
- Customer behavior understanding
- Inventory management optimization
- Personalized customer service approaches

## Usage
1. Ensure all required libraries are installed
2. Load the Mall_Customers.csv dataset
3. Run the clustering analysis
4. Interpret the visualizations and results for business insights

## Future Improvements
- Include additional customer features for more detailed segmentation
- Implement other clustering algorithms for comparison
- Add interactive visualization capabilities
- Develop real-time customer classification system
