import pandas as pd
import streamlit as st
import os
import joblib

st.set_page_config(
    page_title="Loan Approval Checker",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 Loan Approval Checker")

MODEL_PATH = "svc_model/loan_prediction.joblib"

# If model missing -> show waiting message and stop
if not os.path.exists(MODEL_PATH):
    """
    Show warning message and stop the app if the model file is not found.
    """
    st.warning("The system is initializing, please wait")
    st.error("The model file was not found. Please train the model and save it before running the app.")
    st.stop()


@st.cache_resource
def load_model(path: str):
    """Load the model from the given path."""
    return joblib.load(path)

model_data = load_model(MODEL_PATH)
model = model_data["pipeline"]
accuracy = model_data["accuracy"]

# ---- Accuracy Check ----
acc_check = st.button("Check Accuracy")
if acc_check:
    # Using the saved accuracy from the joblib file
    st.info(f"Model Accuracy: {accuracy:.2%}")

# ---- Input form (screenshot-friendly UI) ----
with st.form("loan_form"):
    st.subheader("Loan Application Details")

    applicant_income = st.number_input(
        "Applicant Income (monthly)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income (monthly)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Requested Loan Amount",
        min_value=0.0,
        value=150.0,
        step=10.0
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1.0,
        value=360.0,
        step=12.0
    )

    credit_history = st.selectbox(
        "Credit History",
        options=[1, 0],
        index=0,
        format_func=lambda x: "Exists (1)" if x == 1 else "Does not exist (0)"
    )
    dict_opt = {"Yes": "Married", "No": "Single"}
    married = st.selectbox(
        "Marital Status",
        options=["Yes", "No"],
        index=0,
        format_func=lambda x: dict_opt[x]
    )

    submitted = st.form_submit_button("Check Loan Eligibility")

# Runs only after clicking the button
if submitted:
    status_placeholder = st.empty()
    status_placeholder.warning("Please wait while we check your loan eligibility...")
    # IMPORTANT:
    # Column names must exactly match those used during model training
    X = pd.DataFrame([{
        "Married": married,
        "ApplicantIncome": float(applicant_income),
        "CoapplicantIncome": float(coapplicant_income),
        "LoanAmount": float(loan_amount),
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": int(credit_history),
    }])
    prediction = model.predict(X)[0]
    status_placeholder.empty()
    st.success("Loan Approved") if prediction == "Y" or prediction == 1 else st.error("Loan Rejected")
    # Log the prediction
    log_entry = X.copy()
    log_entry["Loan_Status"] = prediction
    log_entry["Time_Stamp"] = pd.Timestamp.now()
    log_entry.to_csv("Loan_Log.csv", mode='a', header=not os.path.exists("Loan_Log.csv"))
    