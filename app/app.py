import joblib
from fastapi import FastAPI, Body

model = joblib.load("../models/model.pkl")
server = FastAPI()

@server.get("/")
def main():
    return {"message": "Home page of Loan Approval Prediction API"}

@server.get("/health")
def health():
    return {"status": "API is healthy and running"}

# Load trained model

@server.get("/")
def home():
    return {"message": "Loan Prediction API running"}

@server.post("/predict")
def predict(data = Body(...)):
    """
    Expected input format:
    [[1, 0, 2.0, 1, 18500.0, 5200.0, 210.0, 240.0, 1.0, 0, 1, 0]]
    """

    prediction = model.predict(data)
    probability = model.predict_proba(data)
 
    return {
        "prediction": int(prediction[0]),
        "approval_probability": float(probability[0][1])
    }
