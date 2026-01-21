# 🏦 Loan Approval Prediction System

This project is a Machine Learning application developed for my AI/Python course. It predicts whether a loan application will be approved based on user data such as income, marital status, and credit history.

## 🚀 Features
* **Machine Learning Pipeline:** Uses Scikit-Learn `Pipeline` for automated data preprocessing.
* **Support Vector Classifier (SVC):** Implements an RBF kernel model for high-accuracy classification.
* **Interactive GUI:** Built with **Streamlit** for a seamless user experience.
* **Robust Data Handling:** Automated imputation of missing values and feature scaling.

## 🛠️ Tech Stack
* **Language:** Python
* **Library:** Scikit-Learn (Modeling), Pandas (Data Manipulation)
* **GUI:** Streamlit
* **Model Serialization:** Joblib

## 📊 Model Performance
The model was evaluated using 5-fold cross-validation and achieved:
* **Accuracy:** 80.69%
* **Preprocessing:** `StandardScaler` for numeric values and `OneHotEncoder` for categories.

## 🏃 How to Run
   git clone <your-repository-link>

## Install dependencies:
    pip install -r requirements.txt

## Run the app:
    streamlit run main.py
