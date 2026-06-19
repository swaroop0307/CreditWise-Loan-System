"""
predict.py  –  Called by server.js via child_process.spawn
Usage:  python3 predict.py '<json_string>'
Output: JSON printed to stdout  { status, approved, confidence }
"""

import sys
import json
import warnings
import pandas as pd
import joblib

warnings.filterwarnings("ignore")


def load_artifacts():
    """Load model, scaler and encoders saved by trainmodel.py."""
    model    = joblib.load('nb_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, scaler, encoders


def build_feature_row(data: dict, encoders: dict) -> pd.DataFrame:
    """
    Transform the raw string values from the web form into the numeric
    feature vector the model was trained on.
    """
    CATEGORICAL_COLS = [
        'Employment_Status', 'Marital_Status', 'Loan_Purpose',
        'Property_Area', 'Education_Level', 'Gender', 'Employer_Category'
    ]

    row = {
        'Applicant_Income':    float(data['Applicant_Income']),
        'Coapplicant_Income':  float(data['Coapplicant_Income']),
        'Age':                 float(data['Age']),
        'Dependents':          float(data['Dependents']),
        'Credit_Score':        float(data['Credit_Score']),
        'Existing_Loans':      float(data['Existing_Loans']),
        'DTI_Ratio':           float(data['DTI_Ratio']),
        'Savings':             float(data['Savings']),
        'Collateral_Value':    float(data['Collateral_Value']),
        'Loan_Amount':         float(data['Loan_Amount']),
        'Loan_Term':           float(data['Loan_Term']),
    }

    for col in CATEGORICAL_COLS:
        le  = encoders[col]
        val = data[col]
        # Guard: handle unseen labels gracefully
        if val not in le.classes_:
            raise ValueError(
                f"Unknown value '{val}' for field '{col}'. "
                f"Expected one of: {list(le.classes_)}"
            )
        row[col] = int(le.transform([val])[0])

    return pd.DataFrame([row])


def main():
    # ── 1. Parse input ─────────────────────────────────────────────────────────
    if len(sys.argv) < 2:
        print(json.dumps({'status': 'error', 'message': 'No input data provided.'}))
        sys.exit(1)

    try:
        data = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({'status': 'error', 'message': f'JSON parse error: {e}'}))
        sys.exit(1)

    # ── 2. Load model artefacts ────────────────────────────────────────────────
    try:
        model, scaler, encoders = load_artifacts()
    except FileNotFoundError as e:
        print(json.dumps({
            'status':  'error',
            'message': f'Model file not found: {e}. Run trainmodel.py first.'
        }))
        sys.exit(1)

    # ── 3. Build feature row ───────────────────────────────────────────────────
    try:
        df = build_feature_row(data, encoders)
    except (ValueError, KeyError) as e:
        print(json.dumps({'status': 'error', 'message': str(e)}))
        sys.exit(1)

    # ── 4. Scale → Predict ─────────────────────────────────────────────────────
    scaled     = scaler.transform(df)
    prediction = int(model.predict(scaled)[0])
    proba      = model.predict_proba(scaled)[0]

    # Determine which class index = "approved"
    # trainmodel.py uses LabelEncoder on Loan_Approved:
    #   'No'  → 0,  'Yes' → 1   (alphabetical order)
    # If your training used 0/1 integers directly, the encoder may not exist.
    loan_enc = encoders.get('Loan_Approved')
    if loan_enc is not None:
        classes = list(loan_enc.classes_)
        # Accept both string ('Yes'/'No') and int (1/0) encoded targets
        yes_candidates = ['Yes', 'yes', '1', 1]
        approved_idx   = next(
            (i for i, c in enumerate(classes) if c in yes_candidates),
            1   # fallback: assume index 1 = approved
        )
    else:
        # No encoder saved for target → assume 1 = approved
        approved_idx = 1

    is_approved = 1 if prediction == approved_idx else 0
    confidence  = round(float(proba[prediction]) * 100, 2)

    # ── 5. Output ──────────────────────────────────────────────────────────────
    print(json.dumps({
        'status':     'success',
        'approved':   is_approved,
        'confidence': confidence
    }))


if __name__ == '__main__':
    main()
