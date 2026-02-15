import streamlit as st

st.title("Machine Learning Assignment - Credit Card Default Prediction App")
st.write("Streamlit application initial setup")
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

st.title("Credit Card Default Prediction")

st.write("Upload test CSV file (same structure as dataset)")

file = st.file_uploader("Upload CSV", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

if file:
    data = pd.read_csv(file)

    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load(f"model/{model_name}.pkl")

    if "default_payment_next_month" in data.columns:
        y_true = data["default_payment_next_month"]
        X = data.drop("default_payment_next_month", axis=1)
    else:
        X = data
        y_true = None

    if model_name in ["logistic", "knn", "naive_bayes"]:
        X = scaler.transform(X)

    preds = model.predict(X)

    st.subheader("----->Predictions<---")
    st.write(preds)

    if y_true is not None:
        st.subheader("----->Confusion Matrix<---")
        st.write(confusion_matrix(y_true, preds))

        st.subheader("----->Classification Report<---")
        st.text(classification_report(y_true, preds))
