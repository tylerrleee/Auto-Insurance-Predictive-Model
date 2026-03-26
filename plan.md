# Insurance Claims Prediction: Full Project Workflow Plan

## Tailored for State Farm — Rating & Underwriting Modeling Team

---

## 1. Project Overview

**Problem Statement:** Build a classification model that predicts whether an insurance policyholder will file a claim, directly supporting the core work of State Farm's Property & Casualty Actuarial & Underwriting Modeling Department — pricing accuracy and underwriting risk decisions.

**Key Challenge:** The dataset exhibits significant class imbalance (far more non-claim records than claim records), which mirrors the real-world distribution insurers face daily. The project must demonstrate techniques to handle this imbalance while producing models that are actionable for business stakeholders.

**Dataset:** Kaggle — *Insurance Claims Prediction* dataset (historical policyholder data with demographics, claim history, policy details, risk factors, and external factors).

**Language & Tools:** Python (end-to-end)

---

## 2. Phase 1 — Data Acquisition & Initial Exploration

### 2.1 Load and Profile the Data

```
Tools: pandas, numpy
```

- Download the dataset from Kaggle and load into a pandas DataFrame.
- Run `df.info()`, `df.describe()`, and `df.shape` to understand dimensions, dtypes, and memory usage.
- Identify the **target variable** (claim filed: yes/no) and confirm the class distribution — quantify the imbalance ratio (e.g., 90/10 or 95/5).
- Catalog features into groups matching the dataset description:
  - **Policyholder info:** age, gender, occupation, marital status, location
  - **Claim history:** past claim amounts, claim types, frequency, durations
  - **Policy details:** coverage type, policy duration, premium, deductibles
  - **Risk factors:** credit score, driving record, health status, property characteristics
  - **External factors:** economic indicators, weather, regulatory variables

### 2.2 Missing Data Audit

- Compute missing-value percentages per column.
- Classify missingness pattern: MCAR, MAR, or MNAR.
- Decision point: drop columns with >60% missing; flag columns with 10–60% for imputation strategy in Phase 2.

### 2.3 Exploratory Data Analysis (EDA)

```
Tools: matplotlib, seaborn, plotly
```

- **Target distribution:** Bar chart showing claim vs. no-claim counts with exact percentages.
- **Univariate analysis:** Histograms/KDE for continuous features; bar charts for categoricals — all split by the target class.
- **Bivariate analysis:** Correlation heatmap (continuous features), chi-squared tests (categorical vs. target), point-biserial correlation (continuous vs. binary target).
- **Geospatial patterns:** If location data permits, map claim rates by region to identify geographic risk clusters.
- **Claim history deep-dive:** Distribution of past claim frequency and amounts — these are likely the strongest predictors and deserve dedicated attention.

**Deliverable:** A Jupyter notebook section with clearly narrated EDA, suitable for presenting to non-technical stakeholders (maps to the State Farm responsibility of communicating with business partners).

---

## 3. Phase 2 — Data Preparation & Feature Engineering

### 3.1 Data Cleaning

```
Tools: pandas, scikit-learn (SimpleImputer, KNNImputer)
```

- **Missing values:** Median imputation for skewed numerics, mode for categoricals, KNN imputation for features with complex relationships.
- **Outlier treatment:** Use IQR method and domain logic (e.g., a $500K claim on a $50K car warrants investigation, not automatic removal).
- **Duplicates:** Check for duplicate policyholder records; deduplicate or flag.

### 3.2 Feature Engineering

This is where domain knowledge about insurance creates differentiation:

- **Interaction features:**
  - `claim_frequency_per_year` = total past claims / policy duration
  - `premium_to_coverage_ratio` = premium amount / coverage amount (proxy for risk loading)
  - `claims_cost_ratio` = total past claim amounts / total premiums paid
- **Binning with business logic:**
  - Age groups aligned with actuarial brackets (18–25 high-risk, 26–35 moderate, etc.)
  - Credit score tiers (poor/fair/good/excellent)
- **Encoding categoricals:**
  - One-hot encoding for low-cardinality features (gender, marital status)
  - Target encoding for high-cardinality features (occupation, location) — use cross-validated target encoding to avoid leakage
- **Temporal features:** If policy start dates exist, extract tenure, season of policy inception, etc.
- **External factor transforms:** Standardize economic indicators; consider lagging weather variables if temporal alignment allows.

### 3.3 Train/Test Split

```
Tools: scikit-learn (train_test_split, StratifiedKFold)
```

- **Stratified split:** 80/20 train/test with stratification on the target to preserve class ratios.
- **Hold out the test set entirely** — all imputation, scaling, and encoding fit only on training data.
- **Cross-validation strategy:** Stratified 5-fold CV on the training set for model selection.

### 3.4 Feature Scaling

- StandardScaler for linear/distance-based models.
- Tree-based models (the likely winners) don't require scaling, but keep the pipeline flexible.

---

## 4. Phase 3 — Handling Class Imbalance

This is the central technical challenge. Implement and compare multiple strategies:

### 4.1 Resampling Techniques

```
Tools: imbalanced-learn (imblearn)
```

- **SMOTE** (Synthetic Minority Oversampling Technique): Generate synthetic minority-class samples. Test standard SMOTE and variants (BorderlineSMOTE, ADASYN).
- **Random undersampling:** Reduce majority class. Fast but loses information.
- **Combination:** SMOTE + Tomek Links (oversample minority, then clean decision boundaries).

### 4.2 Algorithmic Approaches

- **Class weights:** Pass `class_weight='balanced'` to models that support it (logistic regression, random forest, XGBoost's `scale_pos_weight`).
- **Threshold tuning:** Instead of the default 0.5 cutoff, optimize the classification threshold using precision-recall curves.

### 4.3 Evaluation Metrics Aligned to Business Goals

**Critical decision:** Accuracy is misleading with imbalanced data. Focus on:

- **Primary metric:** F1-Score (harmonic mean of precision and recall) — balances false positives (insuring too cautiously) and false negatives (missing risky policyholders).
- **AUC-ROC:** Measures ranking ability across all thresholds.
- **AUC-PR (Precision-Recall Curve):** More informative than ROC when the positive class is rare.
- **Business-oriented metrics:** Compute the estimated dollar impact of false negatives (missed claims) vs. false positives (over-pricing) to frame results in underwriting terms.

---

## 5. Phase 4 — Model Development & Selection

### 5.1 Baseline Model

```
Tools: scikit-learn (LogisticRegression)
```

- Logistic Regression with `class_weight='balanced'` as the interpretable baseline.
- Record all metrics. This is the "beat this" benchmark.

### 5.2 Tree-Based Ensembles

```
Tools: scikit-learn (RandomForestClassifier, GradientBoostingClassifier), XGBoost, LightGBM
```

- **Random Forest:** Good out-of-the-box; set `class_weight='balanced_subsample'`.
- **XGBoost:** Tune `scale_pos_weight`, `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `subsample`, `colsample_bytree`.
- **LightGBM:** Often faster than XGBoost on larger datasets; tune `is_unbalance` or `scale_pos_weight`.

### 5.3 Hyperparameter Tuning

```
Tools: scikit-learn (RandomizedSearchCV), optuna
```

- Use Optuna for Bayesian hyperparameter optimization (more efficient than grid search).
- Optimize on F1-Score within stratified 5-fold CV.
- Log all experiments for reproducibility.

### 5.4 Stacking Ensemble (Optional, Advanced)

- Combine top 2–3 models using a meta-learner (logistic regression on base model predictions).
- Evaluate whether the stacked model materially improves over the best single model.

---

## 6. Phase 5 — Model Validation & Interpretability

### 6.1 Test Set Evaluation

- Run the final model on the held-out 20% test set (used only once).
- Generate: confusion matrix, classification report, ROC curve, precision-recall curve.
- Compare performance across all imbalance-handling strategies.

### 6.2 Model Interpretability

```
Tools: shap, matplotlib
```

- **SHAP (SHapley Additive exPlanations):**
  - Summary plot: Which features drive predictions overall?
  - Dependence plots: How does each key feature affect claim probability?
  - Force plots: Explain individual predictions (e.g., "Why was this policyholder flagged as high-risk?").
- **Feature importance ranking:** Compare SHAP-based importance to the model's native feature importances.
- **Partial Dependence Plots (PDPs):** Show the marginal effect of features like age, credit score, and claim history on predicted probability — ideal for non-technical presentation.

### 6.3 Model Fairness Check

- Examine whether the model's error rates differ significantly across demographic groups (age, gender, location).
- Document any disparities and recommend mitigations — relevant to regulatory considerations in insurance.

---

## 7. Phase 6 — Business Translation & Recommendations

This phase maps directly to State Farm's emphasis on consulting with business partners and driving data-informed decisions.

### 7.1 Risk Segmentation

- Use the model's predicted probabilities to segment policyholders into risk tiers (Low / Medium / High / Very High).
- Profile each tier: average demographics, claim history, policy characteristics.
- Recommend tier-specific underwriting actions (e.g., additional review for High tier, streamlined approval for Low).

### 7.2 Pricing Implications

- Quantify the expected claim rate per risk tier.
- Estimate the financial impact: "If this model had been used to adjust premiums, the estimated reduction in underwriting loss would be $X."
- Provide a simple cost-benefit analysis of model adoption.

### 7.3 Data Collection Recommendations

- Based on feature importance, identify which data fields are most valuable and recommend prioritizing their collection quality.
- Flag features with high missingness but high predictive value — these represent data collection investment opportunities.
- Recommend any external data sources (e.g., weather APIs, credit bureau feeds) that could improve model performance.

---

## 8. Phase 7 — Documentation & Delivery

### 8.1 Technical Notebook

- Complete Jupyter notebook with narrative markdown cells explaining each decision.
- All code reproducible with a `requirements.txt` and random seeds set.
- Organized with a clear table of contents.

### 8.2 Non-Technical Summary (Stakeholder Deck)

A 5–7 slide summary covering:

1. **The problem:** Why accurate claim prediction matters for pricing and underwriting.
2. **What the data told us:** Key EDA insights in plain language with visuals.
3. **What we built:** Model approach explained without jargon (a "scoring system" that estimates claim likelihood).
4. **How well it works:** Metrics translated to business terms (e.g., "The model correctly identifies 82% of policyholders who will file claims").
5. **What we recommend:** Risk tiers, pricing adjustments, data quality investments.
6. **Next steps:** Production deployment considerations, monitoring plan, model refresh cadence.

### 8.3 GitHub Repository Structure

```
insurance-claims-prediction/
├── README.md                    # Project overview and setup instructions
├── requirements.txt             # Python dependencies
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_preprocessing.ipynb  # Cleaning and feature engineering
│   ├── 03_modeling.ipynb       # Model training and selection
│   └── 04_evaluation.ipynb     # Final evaluation and interpretability
├── src/
│   ├── data_prep.py            # Reusable data preparation functions
│   ├── features.py             # Feature engineering pipeline
│   ├── models.py               # Model training utilities
│   └── evaluation.py           # Evaluation and plotting utilities
├── reports/
│   ├── stakeholder_summary.pdf # Non-technical presentation
│   └── model_card.md           # Model documentation (performance, limitations, fairness)
└── data/
    └── README.md               # Data source and download instructions
```

---

## 9. Python Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
imbalanced-learn>=0.11
shap>=0.43
optuna>=3.4
matplotlib>=3.7
seaborn>=0.13
plotly>=5.18
jupyter>=1.0
```

---

## 10. How This Project Maps to the State Farm Role

| State Farm Responsibility | Project Component |
|---|---|
| Develop and validate advanced analytic models | Phases 4–5: Multiple classifiers with rigorous cross-validation |
| Build datasets to support model development | Phase 2: Feature engineering pipeline with domain-informed features |
| Collaborate on scoping and decision points | Phase 1: Documented decision framework for each analytical choice |
| Present to non-technical business partners | Phase 7: Stakeholder deck translating results to business language |
| Make strategic data collection recommendations | Phase 6.3: Feature importance-driven data investment analysis |
| Work on complex problems of diverse scope | Phase 3: Multiple imbalance strategies compared systematically |
| Peer review and mentoring | Phase 7: Model card and clean code structure enable team review |
| Open-source contribution mindset | Phase 7.3: Public GitHub repo with modular, documented code |
| Business-oriented data science mindset | Phase 6: Risk segmentation, pricing impact, and cost-benefit analysis |