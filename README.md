# Credit Card Default Prediction

## Problem Statement
The objective of this project is to predict whether a credit card customer will default on their payment in the next month based on their historical financial and demographic information.

---

## Dataset Description

- **Dataset Name:** Default of Credit Card Clients  
- **Source:** UCI Machine Learning Repository (via Kaggle)  
- **Dataset Link:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset  
- **Number of Instances:** 30,000  
- **Number of Features:** 23 (excluding ID)  
- **Target Variable:** `default.payment.next.month`  
  - 0 = No Default  
  - 1 = Default  

The dataset contains customer demographic details, credit limit information, repayment history for the last six months, bill statements, and previous payment amounts.

**Note:** The complete dataset was used for model training locally. A sample test dataset is included in this repository for demonstration and Streamlit application usage.


## Models Implemented

The following machine learning classification models were implemented:

1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

---

## Model Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|---------|-----|----------|--------|---------|-----|
| Logistic Regression | 0.81 | 0.77 | 0.68 | 0.36 | 0.47 | 0.39 |
| Decision Tree | 0.73 | 0.66 | 0.45 | 0.45 | 0.45 | 0.28 |
| kNN | 0.78 | 0.70 | 0.55 | 0.30 | 0.39 | 0.30 |
| Naive Bayes | 0.76 | 0.71 | 0.50 | 0.38 | 0.43 | 0.31 |
| Random Forest (Ensemble) | 0.82 | 0.80 | 0.69 | 0.40 | 0.50 | 0.43 |
| XGBoost (Ensemble) | **0.83** | **0.82** | **0.70** | **0.42** | **0.52** | **0.46** |

---

## Model Performance Observations

| ML Model Name | Observation |
|--------------|------------|
| Logistic Regression | Provides a strong baseline performance but struggles with recall due to class imbalance. |
| Decision Tree | Shows lower generalization and tends to overfit the training data, resulting in lower AUC and MCC. |
| kNN | Performance is moderate and sensitive to feature scaling and class distribution. |
| Naive Bayes | Fast and simple model, but independence assumptions limit its predictive performance. |
| Random Forest (Ensemble) | Provides improved stability and better overall performance compared to individual models. |
| XGBoost (Ensemble) | Achieves the best performance across most metrics, making it the most effective model for this dataset. |

---

## Conclusion

Ensemble models significantly outperformed individual models on this dataset.  
**XGBoost** achieved the highest performance in terms of Accuracy, AUC, F1 Score, and MCC, demonstrating its ability to capture complex patterns and handle class imbalance effectively.

---

## Streamlit Application Features

- Upload custom CSV test data  
- Download sample dataset  
- Model selection from multiple trained models  
- Prediction results  
- Evaluation metrics display  
- Confusion matrix  
- Classification report  

---

