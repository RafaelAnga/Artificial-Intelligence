# Market Basket Analysis using Apriori Algorithm
---

## Project Overview
This project implements market basket analysis using the Apriori algorithm to discover associations between products in retail transaction data. The analysis helps identify which products are frequently bought together, enabling better product placement, marketing strategies, and inventory management.

## Technical Stack
- Python 3.x
- Libraries:
  - apyori (for Apriori algorithm implementation)
  - pandas (for data manipulation)
  - numpy (for numerical operations)
  - matplotlib (for visualization)

## Dataset
- File: 'Market_Basket_Optimisation.csv'
- Structure: 7501 transactions with 20 items per transaction
- Format: CSV file with no headers
- Each row represents a single transaction
- Each column contains product names

## Implementation Details
1. Data Preprocessing:
   - Reads transaction data from CSV
   - Converts data into list format required by apriori algorithm
   - Handles string conversion for product names

2. Apriori Algorithm Parameters:
   - min_support = 0.003 (minimum frequency of item occurrence)
   - min_confidence = 0.2 (minimum probability of rule accuracy)
   - min_lift = 3 (minimum strength of association)
   - min_length = 2 (minimum items in a rule)
   - max_length = 2 (maximum items in a rule)

## Analysis Output
The results are presented in a pandas DataFrame with the following columns:
- Left Hand Side: First item in the association rule
- Right Hand Side: Second item in the association rule
- Support: Frequency of items appearing together
- Confidence: Probability of buying second item when first is bought
- Lift: Strength of the association rule

## Key Features
- Automated association rule mining
- Configurable parameters for rule generation
- Results sorted by confidence scores
- Easy visualization of top associations

## Usage
1. Mount Google Drive (if using Google Colab)
2. Load the dataset
3. Run the Apriori algorithm
4. View results sorted by confidence or lift

## Analysis Insights
- Identifies strongest product associations
- Shows purchase patterns
- Helps in cross-selling strategies
- Supports inventory management decisions

---
Note: This implementation is optimized for retail transaction analysis but can be adapted for other association rule mining applications.
