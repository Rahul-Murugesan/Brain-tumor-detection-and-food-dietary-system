import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv('brain_tumor_reports.csv')

# Encode categorical features
le_tumor = LabelEncoder()
df['tumor_type'] = le_tumor.fit_transform(df['tumor_type'])

le_confirmed = LabelEncoder()
df['confirmed'] = le_confirmed.fit_transform(df['confirmed'])  # Yes/No to 1/0

# Features and labels
X = df.drop('confirmed', axis=1)
y = df['confirmed']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = XGBClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
import joblib
joblib.dump(model, 'tumor_confirmation_model.pkl')
joblib.dump(le_tumor, 'tumor_label_encoder.pkl')
joblib.dump(le_confirmed, 'confirmed_label_encoder.pkl')
