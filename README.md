# Predicting Revenue Growth and Cost Reduction from Business AI Adoption

Code accompanying an MSc Business Analytics dissertation, University of Bath, 2026.

This study models revenue growth and cost reduction as separate outcomes through
an identical machine learning pipeline, then compares their SHAP explanations to
identify whether the two are driven by the same or different AI adoption factors.

## Dashboard

Live prototype: [YOUR STREAMLIT URL]

## Contents

| File | Description | Dissertation section |
|---|---|---|
| `notebooks/pipeline.ipynb` | Shared modelling pipeline — preprocessing, training, hyperparameter tuning and evaluation for Linear Regression, Random Forest and XGBoost, run separately on each target | 3.5, 3.6 |
| `notebooks/eda.ipynb` | Exploratory data analysis — data quality checks, distributions, and the target correlation check | 3.3, 4.1 |
| `notebooks/comparative shap analysis.ipynb` | Comparative SHAP analysis across the two outcome models | 3.7, 4.3 |
| `notebooks/subgroup analysis.ipynb` | Predicted outcome variation across industry, company size and region | 4.4 |
| `notebooks/governance shap analysis.ipynb` | Governance-filtered SHAP analysis and cross-method validity check | 4.5 |
| `notebooks/vif check.ipynb` | Variance Inflation Factor analysis | 3.4 |
| `notebooks/residual test.ipynb` | Residual diagnostic supporting the noise-ceiling interpretation | 4.2, 5.2.1 |
| `app.py` | Streamlit dashboard prototype | 3.8, 4.6 |

## Data

The Global AI Adoption and Workforce Impact dataset (150,000 observations,
43 variables) is not included in this repository. It is available from Kaggle:
https://www.kaggle.com/datasets/mohankrishnathalla/global-ai-adoption-and-workforce-impact-dataset

The dataset is synthetic. All results are a demonstration of the modelling
approach rather than empirical claims about real firms.

## Reproducibility

A fixed random seed (42) is applied throughout, ensuring identical train-test
conditions across both target-variable models.

Requirements are listed in `requirements.txt`. The deployed dashboard uses a
5,000-observation sample for SHAP background computation; predictions, input
ranges and uncertainty bands are all derived from the full dataset.
