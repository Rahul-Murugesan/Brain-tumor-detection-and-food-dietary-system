import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# Load saved model, features, and label encoders
model = joblib.load("health_model.pkl")
feature_names = joblib.load("model_features.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Reload data and preprocess it
df = pd.read_csv("chronic_disease_progression.csv")
X = df.drop(columns=['PatientID', 'Date', 'MedicationAdherence', 'StressLevel', 'CognitiveScore', 'MoodScore'])
X = pd.get_dummies(X)
Y = df[['MedicationAdherence', 'StressLevel', 'CognitiveScore', 'MoodScore']]

# Encode targets
for column in Y.columns:
    le = label_encoders[column]
    Y[column] = le.transform(Y[column])

# Align X with the training features
X = X[feature_names]

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Predict
Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

# Evaluation
target_columns = Y.columns

# Accuracy per label
print("🔎 Accuracy Report")
for i, column in enumerate(target_columns):
    train_acc = accuracy_score(Y_train.iloc[:, i], Y_pred_train[:, i])
    test_acc = accuracy_score(Y_test.iloc[:, i], Y_pred_test[:, i])
    print(f"{column} - Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")

# Classification report
print("\n📊 Classification Reports")
for i, column in enumerate(target_columns):
    print(f"\n--- {column} ---")
    print(classification_report(Y_test.iloc[:, i], Y_pred_test[:, i]))

# Confusion matrices
for i, column in enumerate(target_columns):
    cm = confusion_matrix(Y_test.iloc[:, i], Y_pred_test[:, i])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {column}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# ROC Curves
for i, column in enumerate(target_columns):
    fpr, tpr, _ = roc_curve(Y_test.iloc[:, i], model.predict_proba(X_test)[i][:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{column} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for all Labels")
plt.legend()
plt.grid()
plt.show()
