from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and preprocessing artifacts
model = joblib.load('health_model.pkl')
model_features = joblib.load('model_features.pkl')
label_encoders = joblib.load('label_encoders.pkl')

input_fields = [
    'Age', 'Gender', 'Disease', 'MedicalHistory', 'Lifestyle', 'BiomarkerScore',
    'MedicationDose', 'HeartRate', 'BloodPressure_Systolic', 'BloodPressure_Diastolic',
    'Cholesterol', 'BMI', 'SleepHours', 'StepsPerDay', 'Stage', 'Smoker',
    'AlcoholUse', 'SupportSystem', 'HasCaregiver', 'EmploymentStatus'
]

@app.route('/')
def index():
    return render_template('new.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input from form
    input_data = {field: request.form.get(field) for field in input_fields}

    # Convert numeric values
    numeric_fields = ['Age', 'BiomarkerScore', 'MedicationDose', 'HeartRate',
                      'BloodPressure_Systolic', 'BloodPressure_Diastolic',
                      'Cholesterol', 'BMI', 'SleepHours', 'StepsPerDay', 'Stage']
    for field in numeric_fields:
        input_data[field] = float(input_data[field])

    # Create dataframe
    df = pd.DataFrame([input_data])

    # One-hot encoding
    df_encoded = pd.get_dummies(df)

    # Align with training features
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_features]

    # Predict
    preds = model.predict(df_encoded)[0]

    # Decode results
    prediction = {
        label: label_encoders[label].inverse_transform([int(preds[i])])[0]
        for i, label in enumerate(label_encoders)
    }

    return render_template('new.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
