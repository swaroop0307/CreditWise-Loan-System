import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
import joblib

print("⏳ Loading data and training model...")

# Load dataset
df = pd.read_csv('loan_approval_data.csv')

# Drop ID column
if 'Applicant_ID' in df.columns:
    df = df.drop('Applicant_ID', axis=1)

# Replace empty strings with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Fill missing numerical values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill missing categorical values
for col in df.select_dtypes(include=['object', 'string']).columns:
    mode_val = df[col].mode()
    if len(mode_val) > 0:
        df[col] = df[col].fillna(mode_val[0])

# Check remaining NaNs
print("\nMissing values after preprocessing:")
print(df.isnull().sum())

total_nans = df.isnull().sum().sum()
print(f"\nTotal NaNs remaining: {total_nans}")

if total_nans > 0:
    print("\nColumns still containing NaNs:")
    print(df.columns[df.isnull().any()].tolist())
    exit()

# Encode categorical columns
categorical_cols = [
    'Employment_Status',
    'Marital_Status',
    'Loan_Purpose',
    'Property_Area',
    'Education_Level',
    'Gender',
    'Employer_Category',
    'Loan_Approved'
]

encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Features and target
X = df.drop('Loan_Approved', axis=1)
y = df['Loan_Approved']

# Final NaN check
print("NaNs in X:", X.isnull().sum().sum())

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("NaNs after scaling:", np.isnan(X_scaled).sum())

# Train model
nb_model = GaussianNB()
nb_model.fit(X_scaled, y)

# Save files
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')  # remember exact training column order

print("\n✅ Success!")
print("Generated:")
print("- nb_model.pkl")
print("- scaler.pkl")
print("- encoders.pkl")
print("- feature_columns.pkl")