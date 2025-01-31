# Market Basket Analysis using Apriori Algorithm

## Project Overview
This project implements market basket analysis using the Apriori algorithm to discover associations between products in retail transaction data. The analysis helps identify which products are frequently purchased together, enabling better business decisions for product placement and marketing strategies.

## Technical Stack
- Python 3.x
- Libraries:
  - Apyori: For implementing Apriori algorithm
  - Pandas: Data manipulation and analysis
  - NumPy: Numerical operations
  - Matplotlib: Visualization
  - Google Colab: Development environment

## Dataset
The analysis uses 'Market_Basket_Optimisation.csv' containing:
- 7,501 transactions
- 20 items per transaction
- Product names and purchase combinations

## Features
1. Data Preprocessing
   - Transaction data formatting
   - String conversion for product names
   - Data structure optimization

2. Apriori Implementation
   - Parameters:
     - min_support = 0.003 (minimum item frequency)
     - min_confidence = 0.2 (rule strength threshold)
     - min_lift = 3 (association strength threshold)
     - min_length = 2 (minimum items in rule)
     - max_length = 2 (maximum items in rule)

3. Results Analysis
   - Association rule generation
   - Support, Confidence, and Lift calculations
   - Rule visualization in DataFrame format

## Key Metrics
- Support: Frequency of item combinations
- Confidence: Probability of purchasing associated items
- Lift: Strength of item relationships

## Business Applications
- Product placement optimization
- Cross-selling recommendations
- Promotional strategy development
- Inventory management
- Store layout planning
