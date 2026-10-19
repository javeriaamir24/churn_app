from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load saved files (absolute paths work both locally and on PythonAnywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
columns = joblib.load(os.path.join(BASE_DIR, 'columns.pkl'))

# Dropdown fields and their allowed values
CHOICES = {
    'gender': ['Female', 'Male'],
    'SeniorCitizen': ['No', 'Yes'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['No', 'Yes', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'Yes', 'No internet service'],
    'OnlineBackup': ['No', 'Yes', 'No internet service'],
    'DeviceProtection': ['No', 'Yes', 'No internet service'],
    'TechSupport': ['No', 'Yes', 'No internet service'],
    'StreamingTV': ['No', 'Yes', 'No internet service'],
    'StreamingMovies': ['No', 'Yes', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check',
                      'Bank transfer (automatic)', 'Credit card (automatic)'],
}

# Number fields with (minimum, maximum)
NUMERIC = {
    'tenure': (0, 72),
    'MonthlyCharges': (0, 200),
    'TotalCharges': (0, 10000),
}

LABELS = {
    'gender': 'Gender', 'SeniorCitizen': 'Senior Citizen',
    'Partner': 'Has Partner', 'Dependents': 'Has Dependents',
    'PhoneService': 'Phone Service', 'MultipleLines': 'Multiple Lines',
    'InternetService': 'Internet Service', 'OnlineSecurity': 'Online Security',
    'OnlineBackup': 'Online Backup', 'DeviceProtection': 'Device Protection',
    'TechSupport': 'Tech Support', 'StreamingTV': 'Streaming TV',
    'StreamingMovies': 'Streaming Movies', 'Contract': 'Contract Type',
    'PaperlessBilling': 'Paperless Billing', 'PaymentMethod': 'Payment Method',
    'tenure': 'Tenure (months, 0-72)',
    'MonthlyCharges': 'Monthly Charges ($, 0-200)',
    'TotalCharges': 'Total Charges ($, 0-10000)',
}

# Same lists used in Assignment 1
BINARY_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
MULTI_COLS = ['MultipleLines', 'InternetService', 'OnlineSecurity',
              'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']


def validate(form):
    """Check every input. Returns (clean_data, list_of_error_messages)."""
    data, errors = {}, []

    for field, options in CHOICES.items():
        value = form.get(field, '').strip()
        if value == '':
            errors.append(f"Please select a value for {LABELS[field]}.")
        elif value not in options:
            errors.append(f"Invalid choice for {LABELS[field]}.")
        else:
            data[field] = value

    for field, (low, high) in NUMERIC.items():
        raw = form.get(field, '').strip()
        if raw == '':
            errors.append(f"Please enter {LABELS[field]}.")
            continue
        try:
            number = float(raw)
        except ValueError:
            errors.append(f"{LABELS[field]} must be a number.")
            continue
        if not (low <= number <= high):
            errors.append(f"{LABELS[field]} must be between {low} and {high}.")
            continue
        data[field] = number

    return data, errors


def build_features(data):
    """Convert form values into one row that matches the training columns."""
    row = pd.DataFrame(0.0, index=[0], columns=columns)

    row.loc[0, 'SeniorCitizen'] = 1 if data['SeniorCitizen'] == 'Yes' else 0
    row.loc[0, 'gender'] = 1 if data['gender'] == 'Male' else 0
    for col in BINARY_COLS:
        row.loc[0, col] = 1 if data[col] == 'Yes' else 0

    row.loc[0, 'tenure'] = data['tenure']
    row.loc[0, 'MonthlyCharges'] = data['MonthlyCharges']
    row.loc[0, 'TotalCharges'] = data['TotalCharges']
    row.loc[0, 'AvgCostPerMonth'] = data['TotalCharges'] / (data['tenure'] + 1)

    # One-hot encoding: switch on the matching dummy column, if it exists
    for col in MULTI_COLS:
        dummy_name = f"{col}_{data[col]}"
        if dummy_name in row.columns:
            row.loc[0, dummy_name] = 1

    return row


@app.route('/', methods=['GET', 'POST'])
def index():
    result, errors, form = None, [], {}

    if request.method == 'POST':
        form = request.form
        data, errors = validate(form)

        if not errors:
            try:
                features = build_features(data)
                scaled = scaler.transform(features)
                prediction = int(model.predict(scaled)[0])
                probability = float(model.predict_proba(scaled)[0][1]) * 100

                if prediction == 1:
                    result = {'label': 'Churn',
                              'message': 'This customer is likely to leave the company.',
                              'probability': f"{probability:.2f}"}
                else:
                    result = {'label': 'No Churn',
                              'message': 'This customer is likely to stay with the company.',
                              'probability': f"{probability:.2f}"}
            except Exception:
                errors.append("Something went wrong while making the prediction. "
                              "Please check your inputs and try again.")

    return render_template('index.html', choices=CHOICES, numeric=NUMERIC,
                           labels=LABELS, form=form, errors=errors, result=result)


if __name__ == '__main__':
    app.run(debug=True)