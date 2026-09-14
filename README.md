# Telco Customer Churn Prediction: Supervised Machine Learning Pipeline

An end-to-end supervised classification project predicting customer churn using the Telco Customer Churn dataset. The project evaluates linear and ensemble models, establishes reproducible data preprocessing pipelines, and provides an operational blueprint for API deployment, continuous monitoring, and regulatory data governance under Canadian privacy standards (PIPEDA / Law 25).

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset and Canadian Market Context](#dataset-and-canadian-market-context)
- [Pipeline Architecture](#pipeline-architecture)
- [Model Performance and Comparison](#model-performance-and-comparison)
- [Deployment and Monitoring Strategy](#deployment-and-monitoring-strategy)
- [Installation and Execution Guide](#installation-and-execution-guide)
- [Author and Academic Attribution](#author-and-academic-attribution)

---

## Project Overview

Customer acquisition in the telecommunications sector costs significantly more than retaining existing subscribers. This project builds and compares supervised classification models to detect customer churn signals prior to account cancellation. 

The analytical workflow addresses class imbalance, automates missing-value imputation and feature scaling without data leakage, evaluates model tradeoffs using classification metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC), and packages the optimal pipeline for real-time inference.

---

## Repository Structure

    ├── README.md                              <- Project overview, setup, and results
    ├── telco_customer_churn_pipeline.ipynb   <- Google Colab / Jupyter notebook (2-cell cadence)
    ├── report_executive_summary.pdf          <- Two-page formal evaluation and deployment report
    ├── models/
    │   └── canadian_telco_churn_pipeline.joblib <- Serialized scikit-learn pipeline artifact
    ├── data/
    │   └── telco_churn.csv                    <- Local copy of the dataset (optional fallback)
    └── requirements.txt                       <- Environment dependencies

---

## Dataset and Canadian Market Context

* **Source:** IBM / UCI Telco Customer Churn benchmark repository (7,043 subscriber records, 21 initial attributes).
* **Target:** `Churn` (`Yes` = 1, `No` = 0).
* **Class Balance:** 73.5% Retained (`No`), 26.5% Churned (`Yes`).

### Operational Alignment with the Canadian Market
The dataset's behavioral signals directly mirror the Canadian telecommunications landscape (e.g., Rogers, Bell, Telus, and flanker brands like Fido, Virgin Plus, and Koodo):
* **Contract Expiration Cliffs:** Churn vulnerability peaks sharply in months 1–6 and stabilizes after month 24, mirroring customer transition windows following the conclusion of 24-month hardware financing agreements governed by the CRTC Wireless Code.
* **Service Bundling Retention Moats:** Multi-service subscriptions (fiber broadband, online security, streaming, tech support) show strong negative correlation with churn, validating the quad-play bundling strategies used by domestic carriers.
* **Broadband Price-Point Friction:** Subscribers with fiber-optic internet display elevated churn, reflecting promotional discount expiration on premium broadband tiers.

---

## Pipeline Architecture

1. **Ingestion and Data Hygiene:**
   * Removes arbitrary identifiers (`customerID`) to eliminate high-cardinality noise.
   * Converts `TotalCharges` from whitespace objects to numeric types and imputes missing values using the feature median.
2. **Feature Engineering and Leakage Safeguards:**
   * Encodes the binary target (`Churn`).
   * One-hot encodes nominal categorical features with `drop_first=True` to prevent multicollinearity.
   * Partitions data using an 80/20 stratified train/test split.
   * Standardizes numerical attributes (`tenure`, `MonthlyCharges`, `TotalCharges`) via `StandardScaler`, fit strictly on the training partition.
3. **Model Selection:**
   * **Logistic Regression:** Linear baseline with balanced class weighting.
   * **Random Forest Classifier:** Non-linear bagging ensemble (150 estimators, regularized at `max_depth=8`).

---

## Model Performance and Comparison

Models were evaluated on unseen test data (N = 1,409). Given the financial asymmetry between churn outreach and customer attrition, metric analysis prioritizes **Recall** and **ROC-AUC** alongside standard accuracy.

| Model | Accuracy | Precision | Recall (Catch Rate) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 74.2% | 51.1% | **79.4%** | 0.622 | **0.843** |
| **Random Forest** | **78.9%** | **59.2%** | 68.4% | **0.634** | **0.848** |

### Evaluation Takeaways
* **Logistic Regression** is the recommended model for proactive retention outreach. It captures ~80% of churning subscribers (minimizing costly False Negatives), accepting a higher false-alarm rate where promotional retention costs remain low.
* **Random Forest** offers higher precision and fewer false alarms, making it the preferred candidate when intervention costs (e.g., hardware credits) are high and outreach budgets are constrained.

---

## Deployment and Monitoring Strategy

* **Model Serialization:** The data scaler and fitted classifier are bundled into an atomic Scikit-Learn `Pipeline` object exported via `joblib`.
* **API Serving Architecture:** Packaged for containerized serving via **FastAPI** on a serverless container runner (e.g., AWS ECS or Google Cloud Run). Payloads pass through strict Pydantic schema validation to intercept invalid inputs (e.g., negative tenure, null values).
* **Continuous Drift Monitoring:**
  * *Data Drift:* Automated weekly Kolmogorov-Smirnov distribution checks on continuous inputs (`tenure`, `MonthlyCharges`) against baseline training data.
  * *Model Performance Drift:* Monthly evaluation of rolling ROC-AUC against actual billing cycle outcomes. A drop below 0.78 triggers an automated retraining job.
* **Privacy and Governance (PIPEDA / Law 25):**
  * Strips all Personally Identifiable Information (PII) before model scoring.
  * Employs TreeSHAP feature attributions to provide local explainability for automated retention scoring decisions.

---

## Installation and Execution Guide

### Option 1: Run in Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File** > **Open Notebook** > **GitHub** and paste the URL of this repository.
3. Open `telco_customer_churn_pipeline.ipynb` and select **Runtime** > **Run all**.
4. The notebook includes automated fallbacks to fetch the dataset directly from its public repository mirror without manual file uploads.

### Option 2: Local Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/telco-churn-ml-pipeline.git
   cd telco-churn-ml-pipeline
