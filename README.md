# Loan Default Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting whether a borrower will **fully repay a loan** using the **LendingClub loan dataset**. The goal is to classify each loan as either fully paid or not fully paid, based on borrower credit history and loan characteristics.
The project demonstrates a complete pipeline starting from data exploration and encoding to handling class imbalance with SMOTE, and training and evaluating multiple classification models.

---

## 📂 Dataset
The dataset used is the **LendingClub Loan Dataset** (`loan_data.csv`), which contains borrower and loan-level features including:
* `credit.policy` – whether the customer meets LendingClub's credit underwriting criteria
* `purpose` – the purpose of the loan (debt consolidation, credit card, educational, etc.)
* `int.rate` – interest rate of the loan
* `installment` – monthly installment owed
* `log.annual.inc` – natural log of self-reported annual income
* `dti` – debt-to-income ratio
* `fico` – FICO credit score
* `days.with.cr.line` – number of days the borrower has had a credit line
* `revol.bal` – revolving balance
* `revol.util` – revolving line utilization rate
* `inq.last.6mths` – number of credit inquiries in the last 6 months
* `delinq.2yrs` – number of delinquencies in the past 2 years
* `pub.rec` – number of derogatory public records

Target variable:
* `not.fully.paid` – 1 if the loan was not fully paid back, 0 otherwise

---

## ⚙️ Technologies Used
* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* imbalanced-learn (SMOTE)
* XGBoost

---

## 🔄 Workflow

### 1. Data Loading
* Loaded the dataset from `loan_data.csv`
* Inspected structure with `df.head()`, `df.info()`, and `df.describe()`

### 2. Data Exploration
* Checked for missing values across all columns
* Reviewed value counts for every feature to understand distributions and categorical values

### 3. Feature Encoding
* Encoded the categorical `purpose` column into numeric form using `LabelEncoder`

### 4. Data Visualization
* Plotted FICO score distributions split by `credit.policy`
* Plotted FICO score distributions split by loan repayment status (`not.fully.paid`)
* Visualized loan purpose counts split by repayment status
* Used an `lmplot` to explore the relationship between FICO score, interest rate, credit policy, and repayment outcome
* Generated a correlation heatmap across all features

### 5. Train-Test Split
* Split the data into features (`X`) and target (`y`)
* Created an 80/20 train-test split

### 6. Handling Class Imbalance
* Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the minority class (`not.fully.paid = 1`) before training

### 7. Model Building & Training
Trained and compared three classifiers on the SMOTE-resampled data, evaluating each on the original (unbalanced) train and test sets:
* Decision Tree Classifier
* Random Forest Classifier (100 estimators)
* Gradient Boosting Classifier

### 8. Evaluation
For each model:
* Measured train and test **accuracy**
* Generated a confusion matrix (visualized as a heatmap)
* Produced a full classification report (precision, recall, F1-score)

---

## 📊 Results
* Built an end-to-end classification pipeline that addresses a real-world **imbalanced classification problem** (most loans are fully paid, few default)
* Compared 3 ensemble/tree-based models — Decision Tree, Random Forest, and Gradient Boosting — trained on SMOTE-balanced data
* Random Forest and Gradient Boosting generally show stronger and more stable test performance than the single Decision Tree, though results should be interpreted alongside precision/recall for the minority (default) class rather than accuracy alone
* Exact train/test accuracy and classification reports for each model are printed in the notebook

---

## 🚀 How to Run
1. Download the `loan_data.csv` file (LendingClub loan dataset)
2. Place it in the same directory as the notebook, or update the file path in the data-loading cell
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```
4. Run `loan_code.ipynb` top to bottom in Jupyter Notebook or JupyterLab

---

## 📁 Project Structure
```
.
├── loan_code.ipynb   # Main notebook: EDA, encoding, SMOTE, modeling, evaluation
├── loan_data.csv     # Raw loan dataset (not included, download separately)
└── README.md
```

---

## ⭐ Conclusion
This project provides a solid foundation for tackling imbalanced binary classification problems in a financial context, and highlights the importance of resampling techniques like SMOTE alongside careful evaluation using confusion matrices and classification reports rather than accuracy alone.

## 📄 License
Add a license of your choice (e.g. MIT) if you plan to share this publicly.
