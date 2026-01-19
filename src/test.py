import joblib
import warnings
warnings.filterwarnings("ignore")
    # model = joblib.load("C://Users//Amit kumar//Desktop//loan_approval_project//models//model.pkl")
    # print("Pipeline loaded successfully.")
with open("C://Users//Amit kumar//Desktop//loan_approval_project//models//model.pkl", "rb") as f:
    model = joblib.load(f)
    pred = model.predict([[1,0,2.0,1,18500.0,5200.0,210.0,240.0,1.0,0,1,0]])
    print(pred)


# def predict_loan(data):
#     pred = model.predict(data)    
#     prob = model.predict_proba(data)
#     return int(pred[0]), float(prob[0][1])

# predict_loan([[1, 0, 0, 0, 5000, 0, 200, 360, 1, 0, 1, 0]])