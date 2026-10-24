# Week 4: Machine Learning Model Development and Evaluation (Customer Churn)

Virtual Data Science with Python Apprenticeship, Week 4 task.

## Overview
A complete machine learning workflow that predicts whether a telecom customer will churn, from raw data to a business-driven decision threshold. Dataset: IBM Telco Customer Churn (7,043 customers, 26.5% churners).

## Pipeline
1. **Cleaning:** fixed text-typed `TotalCharges` (11 blanks = brand-new customers), dropped customer ID
2. **Split:** stratified 80/20 train/test; test set used once at the end
3. **Preprocessing:** scaling and one-hot encoding inside a scikit-learn Pipeline (no data leakage)
4. **Models:** baseline, logistic regression, decision tree, random forest (all with balanced class weights)
5. **Tuning:** grid search with 5-fold stratified cross-validation (ROC-AUC)
6. **Evaluation:** confusion matrices, ROC and precision-recall curves, learning and complexity curves, bootstrap confidence intervals
7. **Business view:** decision threshold tuned on out-of-fold predictions using an assumed cost of missed churners vs retention offers
8. **Error analysis:** recall and precision by contract, tenure and internet service

## Results (test set)
| Model | ROC-AUC | Recall | Precision |
|-------|---------|--------|-----------|
| Baseline | 0.500 | 0.00 | n/a |
| Logistic Regression | 0.841 | 0.78 | 0.50 |
| Decision Tree | 0.834 | 0.76 | 0.53 |
| Random Forest | 0.844 | 0.79 | 0.53 |

Random forest and logistic regression are statistically tied (AUC difference 95% CI includes 0). With a cost-optimal threshold the model cuts assumed cost by about 25% versus offering everyone a discount.

## Files
- `week4_churn_model.py`: full pipeline script that creates every figure and number
- `telco_churn.csv`: dataset copy
- `Week4_ML_Churn_Model_Report.docx`: full written report

## How to run
```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python week4_churn_model.py
```
