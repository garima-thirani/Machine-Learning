import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("Script started...")

# Creating model folder
os.makedirs("model", exist_ok=True)


# Loading dataset
print("Loading dataset from data folder...")

file_path = "data/UCI_Credit_Card.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
if "ID" in df.columns:
    df.drop("ID", axis=1, inplace=True)

# Target column 
target_column = "default.payment.next.month"

y = df[target_column]
X = df.drop(target_column, axis=1)

# Train test split with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
import pandas as pd

df = pd.read_csv("data/UCI_Credit_Card.csv")

# Drop ID if present
if "ID" in df.columns:
    df = df.drop("ID", axis=1)

# Scaling
scaler = StandardScaler()
X_train_scaled1 = scaler.fit_transform(X_train)
X_test_scaled1 = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# MODELS
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "xgboost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
}

print("\nStarting training the model\n")

for name, model in models.items():
    print(f"Training {name}...")

    if name in ["logistic", "knn", "naive_bayes"]:
        model.fit(X_train_scaled1, y_train)
        y_pred = model.predict(X_test_scaled1)
        y_prob = model.predict_proba(X_test_scaled1)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # Save model
    joblib.dump(model, f"model/{name}.pkl")
    print(f"{name} model trained and saved successfully.")
    print(f"Evaluating {name} model performance...")
    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))
    print("-" * 40)

print("\nAll models trained and saved successfully!")
