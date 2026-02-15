import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    classification_report
)


# Page Configuration
st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="wide",
    page_icon="💳"
)

# Left align tables globally
st.markdown(
    """
    <style>
    table { text-align: left !important; }
    th { text-align: left !important; }
    td { text-align: left !important; }
    </style>
    """,
    unsafe_allow_html=True
)


# Header
st.title("💳 Credit Card Default Prediction System")
st.markdown("""
This application predicts whether a customer will default on their credit card payment.

**Steps to use:**
1. Download the sample dataset (optional)
2. Upload a CSV file
3. Select a model from the sidebar
4. View predictions and performance metrics
""")

st.divider()


# Sidebar (User Flow)
st.sidebar.header("🔧 Controls")

model_name = st.sidebar.selectbox(
    "Step 1: Select Model",
    ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

uploaded_file = st.sidebar.file_uploader(
    "Step 2: Upload Test CSV",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.info("Tip: Use the sample dataset if you don't have a test file.")


# Sample Download
st.subheader("📥 Sample Dataset")

sample_df = pd.read_csv("data/Test_data_credit_card.csv")

st.download_button(
    label="Download Sample Test CSV",
    data=sample_df.to_csv(index=False),
    file_name="test_sample.csv",
    mime="text/csv"
)

st.divider()

# Selected Model Display
st.markdown(f"#### 🤖 Selected Model: **{model_name.replace('_',' ').title()}**")
st.divider()


# Load Model
model = joblib.load(f"model/{model_name}.pkl")
scaler = joblib.load("model/scaler.pkl")
target_col = "default.payment.next.month"


# Main Workflow
if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head(), use_container_width=True)

    # Separate target if exists
    if target_col in data.columns:
        y_true = data[target_col]
        X = data.drop(target_col, axis=1)
    else:
        y_true = None
        X = data

    # Scaling for required models
    if model_name in ["logistic", "knn", "naive_bayes"]:
        X = scaler.transform(X)

    preds = model.predict(X)

    
    # Prediction Summary
    st.subheader("🔍 Prediction Summary")

    pred_counts = pd.DataFrame({
        "Class": [0, 1],
        "Count": [
            (preds == 0).sum(),
            (preds == 1).sum()
        ]
    })

    st.table(pred_counts)

    
    # Metrics Section
    if y_true is not None:

        st.divider()
        st.subheader("📈 Overall Performance Metrics")

        # AUC
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y_true, prob)
        else:
            auc = None

        metrics_df = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC",
                "AUC"
            ],
            "Value": [
                round(accuracy_score(y_true, preds), 4),
                round(precision_score(y_true, preds), 4),
                round(recall_score(y_true, preds), 4),
                round(f1_score(y_true, preds), 4),
                round(matthews_corrcoef(y_true, preds), 4),
                round(auc, 4) if auc else "N/A"
            ]
        })

        st.table(metrics_df)

        
        # Confusion Matrix Table
        st.subheader("📉 Confusion Matrix")

        cm = confusion_matrix(y_true, preds)
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted 0", "Predicted 1"],
            index=["Actual 0", "Actual 1"]
        )

        st.table(cm_df)

        
        # Classification Report Table
        st.subheader("📄 Classification Report")

        report_dict = classification_report(y_true, preds, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        # Round values for cleaner display
        report_df = report_df.round(4)

        st.table(report_df)

else:
    st.info("⬅️ Upload a CSV file in the sidebar to run predictions.")
