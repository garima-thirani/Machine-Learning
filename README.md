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
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | 0.7195 | 0.6091 | 0.3766 | 0.4092 | 0.3922 | 0.2106 |
| kNN | 0.7928 | 0.7013 | 0.5487 | 0.3564 | 0.4322 | 0.3233 |
| Naive Bayes | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| Random Forest (Ensemble) | 0.8112 | 0.7526 | 0.6270 | 0.3610 | 0.4582 | 0.3735 |
| XGBoost (Ensemble) | **0.8117** | **0.7582** | **0.6274** | **0.3655** | **0.4619** | **0.3764** |

---

## Model Performance Observations

| ML Model Name | Observation |
|--------------|------------|
| Logistic Regression | Acts as a baseline model with good precision, but recall is low, indicating difficulty in identifying default cases. |
| Decision Tree | Shows lower overall performance and generalization ability, with the lowest AUC and MCC among all models. |
| kNN | Provides moderate performance but is sensitive to feature scaling and data distribution. |
| Naive Bayes | Achieves higher recall compared to other models, but overall accuracy is moderate due to its strong independence assumptions. |
| Random Forest (Ensemble) | Improves overall stability and performance compared to individual models, with better balance between precision and recall. |
| XGBoost (Ensemble) | Achieves the best overall performance across most metrics, indicating its effectiveness in capturing complex patterns in the dataset. |

---

## Conclusion

Ensemble models (Random Forest and XGBoost) performed better than individual models.  
**XGBoost achieved the highest AUC and MCC**, making it the most effective model for predicting credit card default in this study.


## Streamlit Application Link And Features
- Link : https://credit-card-default-prediction-ml-model.streamlit.app/
- Upload custom CSV test data  
- Download sample dataset  
- Model selection from multiple trained models  
- Prediction results  
- Evaluation metrics display  
- Confusion matrix  
- Classification report  

---

