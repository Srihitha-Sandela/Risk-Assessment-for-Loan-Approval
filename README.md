# Machine Learning–Based Risk Assessment for Loan Approvals
An end-to-end Machine Learning application that predicts loan default risk and supports data-driven loan approval decisions using applicant financial, employment, and credit information.


## Project Overview
Loan approval is a critical process for financial institutions, where inaccurate decisions can result in significant losses. This project leverages Machine Learning to assess an applicant's credit risk and classify them into actionable risk categories.

The solution includes data preprocessing, feature engineering, predictive modeling using LightGBM, and an interactive Streamlit application for real-time risk assessment.
---

## Business Problem
Traditional loan evaluation methods often rely on manual reviews and fixed rules, making them time-consuming and inconsistent.

Loan providers face two major risks:
- Rejecting a creditworthy applicant results in lost business
- Approving a risky applicant can lead to financial loss

This project aims to:
- Predict the probability of loan default
- Identify high-risk applicants
- Support faster and more reliable lending decisions
- Improve credit risk management through data-driven insights 
---

## Dataset Description
- `application_data.csv`: Applicant demographic, financial, and credit-related information
- `previous_application.csv`: Historical loan application behavior
- `columns_description.csv`: Data dictionary describing all variables
---

## Project Workflow
1. Exploratory Data Analysis (EDA)
2. Feature Engineering (behavioral aggregation)
3. Model Building (LightGBM)
4. Model Evaluation
5. Web Application for Loan Eligibility Prediction
---

## Model Used
- **LightGBM Classifier**
  - Fast training
  - Handles missing values well
  - Suitable for large tabular financial data
---

##Tech Stack
Languages and Libraries:
- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- Matplotlib
- Seaborn

Deployment:
- Streamlit

Model Persistence:
- Joblib
---

## Key Insights
- Credit history variables are the strongest predictors of default
- Applicants with multiple previous refusals show higher risk
- Income and age influence default probability but are not sufficient alone
---

## Web Application
A Streamlit-based web interface allows users to input applicant details and receive:
- Loan eligibility decision
- Risk probability score
---

## How to Run
```bash
pip install -r requirements.txt
streamlit run app/app.py
