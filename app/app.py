# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # -------------------------------
# # Page Configuration
# # -------------------------------
# st.set_page_config(
#     page_title="Loan Risk Assessment",
#     page_icon="💳",
#     layout="centered"
# )

# # -------------------------------
# # Load Model & Feature Schema
# # -------------------------------
# @st.cache_resource
# def load_artifacts():
#     model = joblib.load("models/loan_risk_model.pkl")
#     feature_columns = joblib.load("models/feature_columns.pkl")
#     return model, feature_columns

# model, feature_columns = load_artifacts()

# # -------------------------------
# # Title Section
# # -------------------------------
# st.markdown(
#     """
#     <h1 style='text-align: center;'>🏦 Loan Risk Assessment System</h1>
#     <p style='text-align: center; color: gray;'>
#     Machine Learning–based Credit Risk Evaluation
#     </p>
#     """,
#     unsafe_allow_html=True
# )

# st.divider()

# # -------------------------------
# # User Input Section
# # -------------------------------
# st.subheader("📋 Applicant Details")

# col1, col2 = st.columns(2)

# with col1:
#     age = st.number_input("Age (Years)", 18, 70, 30)
#     income = st.number_input("Annual Income", min_value=10000, step=5000)
#     employment_years = st.number_input("Employment Experience (Years)", 0, 40, 5)
#     education = st.selectbox(
#         "Education Level",
#         ["Secondary", "Higher", "Incomplete Higher", "Lower Secondary"]
#     )

# with col2:
#     loan_amount = st.number_input("Requested Loan Amount", min_value=10000, step=5000)
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     employment_type = st.selectbox(
#         "Employment Type",
#         ["Working", "Commercial associate", "State servant", "Self-employed"]
#     )
#     housing = st.selectbox(
#         "Housing Type",
#         ["House / Apartment", "With parents", "Rented apartment"]
#     )

# st.subheader("📊 Credit Profile")

# ext_source = st.slider(
#     "Credit Score (External Source Proxy)",
#     min_value=0.0,
#     max_value=1.0,
#     value=0.5,
#     step=0.01
# )

# # -------------------------------
# # Prediction Button
# # -------------------------------
# st.divider()

# if st.button("🔍 Check Loan Eligibility", use_container_width=True):

#         # ---------------------------------
#     # Business Rule Validation
#     # ---------------------------------
#     if loan_amount > 20 * income:
#         st.error(
#             "❌ Loan amount exceeds acceptable affordability limits.\n\n"
#             "Based on basic risk policy, the requested loan is too high "
#             "relative to the applicant's income."
#         )
#         st.stop()
#     if age < 18 or age > 65:
#         st.error(
#             "❌ Applicant does not meet the age criteria for loan approval.\n\n"
#             "Applicants must be between 18 and 65 years old."
#         )
#         st.stop()
#     if employment_years < 1:
#         st.error(
#             "❌ Insufficient employment history for loan approval.\n\n"
#             "Applicants must have at least 1 year of employment experience."
#         )
#         st.stop()
        

#     # ---------------------------------
#     # Build FULL feature vector
#     # ---------------------------------
#     # final_df = pd.DataFrame(columns=feature_columns)
#     final_df = pd.DataFrame(columns=model.feature_name_)
#     final_df.loc[0] = 0


#     # final_df.loc[0] = 0  # initialize all features with 0

#     # Fill numeric features
#     final_df.at[0, "AGE_YEARS"] = age
#     final_df.at[0, "AMT_INCOME_TOTAL"] = income
#     final_df.at[0, "AMT_CREDIT"] = loan_amount
#     final_df.at[0, "EMPLOYMENT_YEARS"] = employment_years

#     final_df.at[0, "EXT_SOURCE_1"] = ext_source
#     final_df.at[0, "EXT_SOURCE_2"] = ext_source
#     final_df.at[0, "EXT_SOURCE_3"] = ext_source

#     final_df.at[0, "INCOME_CREDIT_RATIO"] = income / loan_amount
#     final_df.at[0, "MISSING_EXT_SOURCE"] = 0

#     # One-hot categorical features (set only if column exists)
#     gender_col = f"CODE_GENDER_{gender}"
#     if gender_col in final_df.columns:
#         final_df.at[0, gender_col] = 1

#     education_col = f"NAME_EDUCATION_TYPE_{education}"
#     if education_col in final_df.columns:
#         final_df.at[0, education_col] = 1

#     income_type_col = f"NAME_INCOME_TYPE_{employment_type}"
#     if income_type_col in final_df.columns:
#         final_df.at[0, income_type_col] = 1

#     housing_col = f"NAME_HOUSING_TYPE_{housing}"
#     if housing_col in final_df.columns:
#         final_df.at[0, housing_col] = 1

#     # ---------------------------------
#     # Predict
#     # ---------------------------------
#     prob = model.predict_proba(final_df)[0][1]

#     # Risk Decision
#     if prob < 0.25:
#         decision = "✅ Approved – Low Risk"
#         color = "green"
#     elif prob < 0.50:
#         decision = "⚠️ Approved – Medium Risk (Higher Interest)"
#         color = "orange"
#     else:
#         decision = "❌ Rejected – High Risk"
#         color = "red"

#     # -------------------------------
#     # Results Section
#     # -------------------------------
#     st.divider()
#     st.subheader("📌 Loan Decision Result")

#     st.markdown(
#         f"""
#         <h2 style='color:{color}; text-align:center;'>{decision}</h2>
#         <p style='text-align:center; font-size:18px;'>
#         Risk Probability: <b>{prob:.2%}</b>
#         </p>
#         """,
#         unsafe_allow_html=True
#     )

#     st.info(
#         "⚠️ This tool is a decision-support system and does not replace human judgment."
#     )

# # -------------------------------
# # Footer
# # -------------------------------
# st.divider()
# st.caption(
#     "© 2025 | Machine Learning-Based Risk Assessment for Loan Approvals"
# )

import streamlit as st
import pandas as pd
import joblib

# ======================================================
# Page Configuration
# ======================================================
st.set_page_config(
    page_title="RiskLens – Loan Guardian",
    page_icon="🏦",
    layout="wide"
)

# ======================================================
# Custom CSS
# ======================================================
st.markdown("""
<style>
html, body {
    font-family: 'Inter', system-ui, -apple-system;
    background-color: #F6F9FC;
    color: #0F172A;
}
.block-container {
    padding-top: 0rem;
    max-width: 1350px;
}
.hero {
    background: linear-gradient(135deg, #0B3C5D, #0F4C75);
    padding: 55px 70px;
    border-radius: 0 0 32px 32px;
    margin-bottom: 45px;
}
.hero h1 {
    font-size: 44px;
    font-weight: 800;
    color: white;
}
.hero p {
    font-size: 16px;
    color: #DCE7F1;
}
.badge {
    background: rgba(255,255,255,0.15);
    color: white;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}
.card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 22px rgba(15,23,42,0.06);
    margin-bottom: 24px;
}
.section-title {
    font-size: 18px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# Load Model + Feature Schema (CRITICAL FIX)
# ======================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/loan_risk_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_artifacts()

# ======================================================
# Session State
# ======================================================
if "result" not in st.session_state:
    st.session_state.result = None

# ======================================================
# HERO
# ======================================================
st.markdown("""
<div class="hero">
    <span class="badge">Enterprise-Grade Credit Risk Assessment</span>
    <h1>ML-Powered Loan <span style="color:#2dd4bf;">Risk Assessment</span></h1>
    <p>ML + policy based credit risk evaluation</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# INPUT PAGE
# ======================================================
if st.session_state.result is None:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card"><div class="section-title">👤 Personal</div>', unsafe_allow_html=True)
        name = st.text_input("Full Name", "John Doe")
        age = st.slider("Age", 18, 70, 30)
        employment_years = st.slider("Employment Years", 0, 40, 5)
        employment_type = st.selectbox(
            "Employment Type",
            ["Working", "Commercial associate", "State servant", "Self-employed"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="section-title">💰 Financial</div>', unsafe_allow_html=True)
        income = st.slider("Annual Income (₹)", 10000, 500000, 50000, step=5000)
        loan_amount = st.slider("Loan Amount (₹)", 10000, 1000000, 200000, step=10000)
        credit_score = st.slider("Credit Score (CIBIL)", 300, 850, 700, step=10)
        housing = st.selectbox(
            "Property Ownership",
            ["House / Apartment", "With parents", "Rented apartment"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Start Assessment →", type="primary", use_container_width=True):

        # ==================================================
        # RAW SIGNALS
        # ==================================================
        monthly_income = income / 12
        estimated_emi = loan_amount / 36
        emi_ratio = estimated_emi / monthly_income

        # ==================================================
        # BUILD FEATURE VECTOR (CORRECT)
        # ==================================================
        # Build inference dataframe using EXACT training schema
        # Build feature vector EXACTLY as model expects
        feature_columns = model.feature_name_
        X = pd.DataFrame(0, columns=feature_columns, index=[0])

        X.at[0, "AGE_YEARS"] = age
        X.at[0, "AMT_INCOME_TOTAL"] = income
        X.at[0, "AMT_CREDIT"] = loan_amount
        X.at[0, "EMPLOYMENT_YEARS"] = employment_years

        credit_norm = (credit_score - 300) / 550
        credit_norm = max(0, min(credit_norm, 1))

        X.at[0, "EXT_SOURCE_1"] = credit_norm
        X.at[0, "EXT_SOURCE_2"] = credit_norm
        X.at[0, "EXT_SOURCE_3"] = credit_norm

        X.at[0, "INCOME_CREDIT_RATIO"] = income / loan_amount
        X.at[0, "MISSING_EXT_SOURCE"] = 0

        for col in [
            f"NAME_INCOME_TYPE_{employment_type}",
            f"NAME_HOUSING_TYPE_{housing}"
        ]:
            if col in X.columns:
                X.at[0, col] = 1

        prob = model.predict_proba(X)[0][1]

        # ==================================================
        # MODEL PREDICTION
        # ==================================================
        prob = model.predict_proba(X)[0][1]

        # ==================================================
        # POST-MODEL RISK ADJUSTMENT (CRITICAL FIX)
        # ==================================================
        final_prob = prob

        if emi_ratio > 0.4:
            final_prob += 0.20
        elif emi_ratio > 0.3:
            final_prob += 0.10

        if credit_score < 600:
            final_prob += 0.15

        if employment_years < 2:
            final_prob += 0.05

        final_prob = min(final_prob, 1.0)

        st.session_state.result = (name, loan_amount, final_prob)
        st.rerun()

# ======================================================
# RESULT PAGE
# ======================================================
else:
    name, loan, final_prob = st.session_state.result

    col1, col2, col3 = st.columns(3)
    col1.metric("Default Risk Probability", f"{final_prob:.2%}")
    col2.metric("Approval Confidence", f"{(1-final_prob):.2%}")

    if final_prob < 0.35:
        risk = "Low Risk 🟢"
        decision = "Approved"
    elif final_prob < 0.60:
        risk = "Medium Risk 🟡"
        decision = "Manual Review"
    else:
        risk = "High Risk 🔴"
        decision = "Rejected"

    col3.metric("Risk Category", risk)

    st.divider()
    st.markdown(f"""
    **Decision:** {decision}  
    **Risk Probability:** {final_prob:.2%}
    """)

    if st.button("Assess Another Application"):
        st.session_state.result = None
        st.rerun()

