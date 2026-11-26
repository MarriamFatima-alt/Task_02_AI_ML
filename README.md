# Task 02 - Build and Compare Machine Learning Models

**Intern:** Marriam Fatima | **Intern ID:** U-54
**Program:** Devixo Solutions AI/ML Internship

## Objective
Build several machine learning models on a real-world dataset, evaluate them with appropriate metrics, compare them and select the best one.

## Dataset
**Telco Customer Churn** (Kaggle): 7,043 customers, 21 columns.
Target: `Churn` (Yes/No), a binary classification problem.
Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Steps
**Part 1 - Data Preprocessing**
- Loaded the dataset
- Fixed `TotalCharges` (text with 11 blank values converted to numbers, blanks filled with 0)
- Removed duplicates (none found) and dropped `customerID`
- Encoded categorical features (One-Hot Encoding; Churn mapped to 0/1)
- Split into 80% train and 20% test (stratified)
- Scaled numerical columns (StandardScaler, fitted on train data only)

**Part 2 and 3 - Models and Evaluation**
Logistic Regression, Decision Tree, Random Forest, KNN, SVM. Evaluated with Accuracy, Precision, Recall, F1 Score and Confusion Matrix.

**Part 4 - Model Comparison**
Comparison table with training time, metrics, advantages and limitations.

**Bonus - Hyperparameter Tuning**
GridSearchCV on Random Forest.

## Results

| Model | Training Time (s) | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Logistic Regression | 0.053 | 0.8062 | 0.6593 | 0.5588 | 0.6049 |
| Random Forest (Tuned) | 0.503 | 0.8062 | 0.6760 | 0.5187 | 0.5870 |
| SVM | 1.290 | 0.7949 | 0.6545 | 0.4813 | 0.5547 |
| KNN | 0.006 | 0.7630 | 0.5538 | 0.5508 | 0.5523 |
| Random Forest | 0.973 | 0.7850 | 0.6187 | 0.4947 | 0.5498 |
| Decision Tree | 0.122 | 0.7253 | 0.4825 | 0.4786 | 0.4805 |

## Conclusion
**Logistic Regression** performed best overall (highest F1 Score and Recall, joint-highest Accuracy) and trained very fast. Hyperparameter tuning improved Random Forest (F1 from 0.5498 to 0.5870), but not enough to beat Logistic Regression. Even the best model misses about 44% of customers who actually churn, partly because the data is imbalanced (about 27% churn).

## Files
- `notebook.ipynb`: complete notebook (code and outputs)
- `dataset.csv`: Telco Customer Churn dataset (original Kaggle file name: WA_Fn-UseC_-Telco-Customer-Churn.csv)
- `Images/`: screenshots of each step
- `Marriam_Fatima_U-54_Task2.pdf`: final report

Note: in the notebook the dataset is loaded as `WA_Fn-UseC_-Telco-Customer-Churn.csv`. To run it again, use `dataset.csv` in its place.
