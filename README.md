# Anomaly and Fraud Detection in Finance

## Overview
This project focuses on detecting credit card fraud using anomaly detection models while exploring model explainability.  
We analyze both **global** (dataset-level) and **local** (instance-level) explanations to understand **feature contributions** and **model behavior**, using **LIME** and **SHAP** frameworks.

---

## Objective
- Detect anomalies (fraudulent transactions) from an imbalanced dataset.
- Assess **global** and **local explainability** of different anomaly detectors.
- Compare explanation consistency across models when they agree or disagree on predictions.

---

## Dataset Description
- **Source**: Credit card transactions made by European cardholders over two days in September 2013.
- **Size**: 284,807 transactions.
- **Fraud Cases**: 492 fraudulent transactions (~0.173% of the dataset).
- **Features**:
  - `Time`: Seconds elapsed since the first transaction.
  - `V1–V28`: Principal Component Analysis (PCA) transformed features.
  - `Amount`: Transaction amount.
  - `Class`: 0 (authentic) or 1 (fraudulent).
- After feature engineering, **34 features** were available.

---

## Feature Selection
To increase computational efficiency, dimensionality was reduced from 34 to 10 features using:
- **Recursive Feature Elimination (RFE)**
- **SelectFromModel (SFM)**
- **BorutaPy**

---

## Models Trained
Six models were trained for fraud detection:
- Logistic Regression
- Random Forest
- Gradient Boosting
- LightGBM Classifier
- K-Nearest Neighbors (KNN) — from PyOD
- Isolation Forest — from PyOD

---

## Model Performance

### ROC-AUC and Training Time
![Model Performance and Training Time](path_to_fig1.png)

- **LightGBM** achieved the best ROC-AUC score and fastest training time.
- **Random Forest** and **Isolation Forest** offered good trade-offs between performance and computational efficiency.
- **KNN** and **Gradient Boosting** had good accuracy but longer training times.

---

## Global Explanation

### 1. Logistic Regression Coefficients
![Feature Importance: Logistic Regression](path_to_fig2.png)

- **Top Contributor**: `V10`
- **Lowest Contributor**: `V17`
- Most features contributed **negatively**, except `V11`.

---

### 2. Feature Importance Comparisons

#### Built-in Model Feature Importances
![Built-in Feature Importances](path_to_fig3.png)

- Tree-based models (Random Forest, Gradient Boosting, LightGBM, Isolation Forest) showed feature importances via **mean decrease in impurity** (Gini importance).

#### Permutation Feature Importances
![Permutation Feature Importances](path_to_fig4.png)

- **Permutation importance** revealed slightly different rankings, identifying `V10`, `V11`, and `V17` as consistently important features.

**Key Insight**:  
Permutation feature importance is more robust for model-agnostic explanations, particularly in handling multicollinearity and nonlinear relationships.

---

## Local Explanation

### Agreement and Disagreement Cases

#### Case 1: All Three Models Agree
- Models: Gradient Boosting, Isolation Forest, LightGBM.
- All correctly predicted a fraudulent transaction.
- Key contributing features: `V10`, `V11`, `V16`, `V17`.
- **LIME and SHAP explanations** were consistent for Isolation Forest but varied slightly across other models.

![LIME and SHAP: All Models Agree](path_to_fig5.png)

---

#### Case 2: Gradient Boosting and Isolation Forest Agree
- Models: Gradient Boosting and Isolation Forest agreed, LightGBM disagreed.
- Feature contributions differed more significantly across models.
- Highlighted model architecture differences.

![LIME and SHAP: GBM and Isolation Forest Agree](path_to_fig6.png)

---

#### Case 3: Isolation Forest and LightGBM Agree
- Models: Isolation Forest and LightGBM agreed, Gradient Boosting disagreed.
- **Feature `V17`** was consistently important across models even when disagreements occurred.

![LIME and SHAP: Isolation Forest and LightGBM Agree](path_to_fig7.png)

---

## Conclusion
- **LightGBM** performed best in terms of predictive performance (ROC-AUC) and training efficiency.
- Global explanations provided insight into feature contributions across models.
- Local explanations showed how model explanations vary even when predictions agree.
- **Permutation feature importance** is generally more robust than built-in feature importance.
- Future work could involve hybrid models and deeper analysis of feature interactions.

---

## Future Work
- Build ensemble methods combining top-performing models.
- Explore deep learning approaches with enhanced interpretability (e.g., DeepSHAP).
- Extend local explanation consistency analysis across larger samples.

---

## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Vinylango25/Anomaly-Fraud-detection-in-Finance.git
    cd Anomaly-Fraud-detection-in-Finance
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Launch Jupyter Notebook**:
    Open and run the notebook `Anomaly_Fraud_Detection_Finance.ipynb`.

---

## Folder Structure
