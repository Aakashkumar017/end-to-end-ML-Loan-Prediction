print("Starting training...")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
import pandas as pd

df = pd.read_csv("./dataset.csv")

y = df["Loan_Status"]
X = df.drop("Loan_Status", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# Manually applying scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)




pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
    ))
])

# Train the full pipeline
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "../models/trained_model.pkl")

print("Model trained and saved successfully")
