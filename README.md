# churn-prediction

End-to-end customer churn prediction with business impact analysis.

---

## Overview

This project predicts which customers are likely to churn using a telecom dataset, and translates model outputs into concrete business metrics — estimated revenue at risk, cost of intervention, and ROI of retention campaigns.

The goal is to go beyond accuracy numbers and answer the question that actually matters to a business: *what does acting on this model cost, and what does it save?*

---

## What's in this repo

```
churn-prediction/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb        # Model training and evaluation
│   └── 04_business_impact.ipynb  # Cost-benefit and threshold analysis
├── src/
│   └── predict.py                # Inference script
├── results/                      # Saved outputs and plots
├── Dockerfile                    # Container definition
├── k8s/                          # Kubernetes deployment manifests
└── requirements.txt
```

---

## Approach

**Data:** IBM Telco Customer Churn dataset (~7,000 customers, 20 features).

**Target:** Binary classification — will this customer churn in the next billing period?

**Models compared:** Logistic Regression, Random Forest, XGBoost. Final model selected based on ROC-AUC with threshold tuned for business cost optimisation rather than raw accuracy.

**Business impact analysis:** Different misclassification errors have different costs:
- False negative (missed churner): lose the customer's lifetime value
- False positive (unnecessary retention offer): cost of the offer

The optimal threshold is where expected savings from interventions exceed their cost. This is calculated explicitly in `04_business_impact.ipynb`.

---

## Key results

| Metric | Value |
|---|---|
| ROC-AUC | 0.85 |
| Precision at optimal threshold | 0.71 |
| Recall at optimal threshold | 0.76 |
| Estimated revenue saved per 1,000 customers | ~$18,400 |

---

## Running locally

**Requirements:** Python 3.10+, or Docker.

```bash
# Option 1: local
git clone https://github.com/valofils/churn-prediction.git
cd churn-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab

# Option 2: Docker
docker build -t churn-prediction .
docker run -p 8888:8888 churn-prediction
```

---

## Deployment

Kubernetes manifests are included in `k8s/` for deploying the inference service to a cluster.

```bash
kubectl apply -f k8s/
```

---

## Tech stack

`Python` `XGBoost` `scikit-learn` `pandas` `matplotlib` `Docker` `Kubernetes`

---

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — publicly available on Kaggle, no API key required.
