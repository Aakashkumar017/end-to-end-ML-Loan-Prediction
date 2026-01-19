# end-to-end-ML-Loan-Prediction
---
loan_approval_project/
│
├── app/
│   ├── __init__.py
│   └── app.py              # FastAPI app for prediction
│
├── data/
│   └── loan_data_2000_with_nulls_and_3plus.csv
│
├── models/
│   └── loan_approval_model.pkl
│
├── notebooks/
│   └── loan.ipynb          # EDA and experimentation
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py         # Preprocessing + ML pipeline
│   ├── train.py            # Model training script
│   └── predict.py          # Inference logic
│
├── requirements.txt
└── README.md
