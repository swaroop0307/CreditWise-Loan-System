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
    model = joblib.load('nb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')

    try:
        feature_columns = joblib.load('feature_columns.pkl')
    except:
        feature_columns = None

    return model, scaler, encoders, feature_columns


def build_feature_row(data: dict, encoders: dict, feature_columns=None):
    """
    Convert web form data into model input.
    """

    categorical_cols = [
        'Employment_Status',
        'Marital_Status',
        'Loan_Purpose',
        'Property_Area',
        'Education_Level',
        'Gender',
        'Employer_Category'
    ]

    row = {
        'Applicant_Income': float(data['Applicant_Income']),
        'Coapplicant_Income': float(data['Coapplicant_Income']),
        'Age': float(data['Age']),
        'Dependents': float(data['Dependents']),
        'Credit_Score': float(data['Credit_Score']),
        'Existing_Loans': float(data['Existing_Loans']),

        # User enters 35, model expects 0.35
        'DTI_Ratio': float(data['DTI_Ratio']) / 100.0,

        'Savings': float(data['Savings']),
        'Collateral_Value': float(data['Collateral_Value']),
        'Loan_Amount': float(data['Loan_Amount']),
        'Loan_Term': float(data['Loan_Term']),
    }

    for col in categorical_cols:

        le = encoders[col]
        value = data[col]

        if value not in le.classes_:
            raise ValueError(
                f"Unknown value '{value}' for field '{col}'. "
                f"Expected one of: {list(le.classes_)}"
            )

        row[col] = int(le.transform([value])[0])

    df = pd.DataFrame([row])

    # Reorder columns to match training order
    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in df.columns]

        if len(missing) > 0:
            raise ValueError(
                f"Missing features: {missing}"
            )

        df = df[feature_columns]

    return df


def main():

    # Parse JSON input
    if len(sys.argv) < 2:
        print(json.dumps({
            'status': 'error',
            'message': 'No input data provided.'
        }))
        sys.exit(1)

    try:
        data = json.loads(sys.argv[1])

    except json.JSONDecodeError as e:

        print(json.dumps({
            'status': 'error',
            'message': f'JSON parse error: {e}'
        }))
        sys.exit(1)

    # Load artifacts
    try:

        model, scaler, encoders, feature_columns = load_artifacts()

    except Exception as e:

        print(json.dumps({
            'status': 'error',
            'message': f'Could not load model files: {e}'
        }))
        sys.exit(1)

    # Build feature row
    try:

        df = build_feature_row(
            data,
            encoders,
            feature_columns
        )

    except Exception as e:

        print(json.dumps({
            'status': 'error',
            'message': str(e)
        }))
        sys.exit(1)

    # Scale features
    try:
        scaled = scaler.transform(df)

    except Exception as e:

        print(json.dumps({
            'status': 'error',
            'message': f'Scaling failed: {e}'
        }))
        sys.exit(1)

    # Predict
    try:

        prediction = int(model.predict(scaled)[0])
        proba = model.predict_proba(scaled)[0]

    except Exception as e:

        print(json.dumps({
            'status': 'error',
            'message': f'Prediction failed: {e}'
        }))
        sys.exit(1)

    # Find approved class
    loan_enc = encoders.get('Loan_Approved')

    if loan_enc is not None:

        classes = list(loan_enc.classes_)

        approved_idx = next(
            (
                i for i, c in enumerate(classes)
                if str(c).lower() == "yes"
            ),
            1
        )

    else:
        approved_idx = 1

    is_approved = 1 if prediction == approved_idx else 0

    confidence = round(
        float(proba[prediction]) * 100,
        2
    )

    print(json.dumps({
        'status': 'success',
        'approved': is_approved,
        'confidence': confidence
    }))


if __name__ == '__main__':
    main()