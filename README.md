# Credit-Building Potential Segmentation

A data science project segmenting borrowers by credit-building potential using
K-means clustering, logistic regression, and decision trees — framed as a consumer
fintech product decision.

---

## Project Overview

This project explores the **Give Me Some Credit** dataset to identify which borrowers
show the strongest potential to successfully build credit, and how a credit-building
fintech product should respond differently to each segment.

The analysis is structured around a core product question: among a population of
150,000 borrowers, who is most likely to benefit from a credit-building product —
and who needs something different first?

**Two analytical approaches:**
1. **Segmentation** — K-means clustering to identify distinct borrower profiles
2. **Modeling** — logistic regression and decision tree to predict delinquency risk

---

## Project Structure

```
credit-readiness-segmentation/
├── data/                  # Not tracked — see Dataset section below
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_segmentation.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_product_framing.ipynb
├── .gitignore
└── README.md
```

---

## Dataset

### Source
- **Dataset:** Give Me Some Credit — Kaggle Competition
- **File used:** `cs-training.csv` only (test file excluded — no target variable)
- **Size:** 150,000 borrowers, 10 features + 1 binary target

Due to Kaggle's redistribution policy, the data is not included in this repository.

**To run this project:**
1. Download `cs-training.csv` from [Kaggle](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)
2. Place it in the `data/` directory
3. Run notebooks in order: `01` → `02` → `03` → `04`

### Raw Features

| Column | Type | Description |
|---|---|---|
| `SeriousDlqin2yrs` | binary | Target: 90+ days past due in last 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | float | Credit card balance as % of credit limit |
| `age` | int | Borrower age in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | int | Times 30–59 days past due in last 2 years |
| `DebtRatio` | float | Monthly debt payments / monthly gross income |
| `MonthlyIncome` | float | Monthly gross income |
| `NumberOfOpenCreditLinesAndLoans` | int | Open loans and lines of credit |
| `NumberOfTimes90DaysLate` | int | Times 90+ days past due |
| `NumberRealEstateLoansOrLines` | int | Mortgage and real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | int | Times 60–89 days past due in last 2 years |
| `NumberOfDependents` | int | Dependents excluding borrower |

### Engineered Features

| Column | Description |
|---|---|
| `MonthlyIncome_log` | log1p(MonthlyIncome) — addresses right skew |
| `DebtRatio_high` | 1 if debt obligations exceed monthly income |
| `EverLate30` | 1 if ever 30–59 days past due |
| `EverLate60` | 1 if ever 60–89 days past due |
| `EverLate90` | 1 if ever 90+ days past due |

---

## Key EDA Findings

- **Target imbalance** — 6.7% positive delinquency rate; class weighting applied in modeling
- **MonthlyIncome** had ~21% missing values (including near-zero data entry errors) — imputed using KNN
- **RevolvingUtilization** bimodal — large group near 0 and spike at 1.0 (maxed out)
- **Past-due columns** highly correlated (0.98–0.99) — binarized to reduce multicollinearity
- **Age** negatively correlated with delinquency — older borrowers slightly less likely to default

---

## Segmentation Results

**k=4 selected** (silhouette score = 0.41)

| Segment | Size | % of Total | Actual Delinquency Rate |
|---|---|---|---|
| Established & Stable | 92,693 | 61.8% | 3.1% |
| Debt Burdened | 30,741 | 20.5% | 2.7% |
| Early Risk | 18,242 | 12.2% | 15.5% |
| Seriously Delinquent | 8,324 | 5.5% | 41.7% |

---

## Modeling Results

### Logistic Regression

| Metric | Score |
|---|---|
| AUC-ROC | 0.866 |
| Precision (class 1) | 0.22 |
| Recall (class 1) | 0.75 |
| F1 (class 1) | 0.35 |
| Accuracy | 0.81 |

### Decision Tree

| Metric | Score |
|---|---|
| AUC-ROC | 0.848 |
| Precision (class 1) | 0.17 |
| Recall (class 1) | 0.86 |
| F1 (class 1) | 0.28 |
| Accuracy | 0.70 |

**Logistic regression selected as primary model** — higher AUC-ROC, better
precision-recall balance, and more interpretable coefficients.

**Top predictors:** RevolvingUtilization, EverLate90, EverLate60, EverLate30, DebtRatio_high

---

## Product Recommendations

| Segment | Recommended Product | Expected Outcome |
|---|---|---|
| **Early Risk** | Credit builder line ($300–$500) + financial education | Score improvement within 6–12 months |
| **Debt Burdened** | Budgeting / debt management tools | Reduced debt burden; credit product in phase 2 |
| **Established & Stable** | Not a primary target | Already credit healthy |
| **Seriously Delinquent** | Debt counseling before any credit product | Stabilization before credit building |

---

## Limitations

- Dataset sourced from a 2011 Kaggle competition — credit behavior patterns may differ in current market
- MonthlyIncome contained significant data quality issues (~21% missing or erroneous)
- `1 - risk_score` potential score is a simplified proxy — production scoring would incorporate business-specific weights validated by repayment outcome data
- Observational data — causal claims require randomized experiments

---

## Tech Stack

- **Analysis:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Modeling:** `scikit-learn`, `statsmodels`

---

## Author

Amanda Tan — [GitHub](https://github.com/amandatan0106)
