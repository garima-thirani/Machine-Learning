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

# CREATING MODEL FOLDER
os.makedirs("model", exist_ok=True)

# LOADING DATASET (UCI CREDIT CARD)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
df = pd.read_excel(url, header=1)

# Clean column names
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace(".", "")

# Drop ID
df.drop("ID", axis=1, inplace=True)

# Split features & target
y = df["default_payment_next_month"]
X = df.drop("default_payment_next_month", axis=1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# MODELS
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "xgboost": XGBClassifier(eval_metric="logloss")
}

print("\n-------MODEL PERFORMANCE------\n")

for name, model in models.items():
    if name in ["logistic", "knn", "naive_bayes"]:
        model.fit(X_train_scale1, y_train)
        y_pred = model.predict(X_test_scaled1)
        y_prob = model.predict_proba(X_test_scaled1)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    joblib.dump(model, f"model/{name}.pkl")

    print(name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))
    print("-" * 30)
