import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# Set seed
np.random.seed(42)

# Create feature data (1000 samples, 6 features)
n_samples = 1000
X = pd.DataFrame({
    'HeartRate': np.random.normal(75, 10, n_samples),
    'SleepHours': np.random.normal(7, 1.5, n_samples),
    'ExerciseMinutes': np.random.normal(30, 10, n_samples),
    'WaterIntake': np.random.normal(2.5, 0.5, n_samples),
    'ScreenTime': np.random.normal(5, 2, n_samples),
    'OutdoorTime': np.random.normal(2, 1, n_samples),
})

# Simulated correlations
Y = pd.DataFrame()
Y['MedicationAdherence'] = ((X['HeartRate'] < 80) & (X['SleepHours'] > 6)).astype(int)
Y['StressLevel'] = ((X['ExerciseMinutes'] < 20) | (X['ScreenTime'] > 6)).astype(int)
Y['CognitiveScore'] = ((X['WaterIntake'] > 2) & (X['SleepHours'] > 7)).astype(int)
Y['MoodScore'] = ((X['OutdoorTime'] > 1.5) & (X['ExerciseMinutes'] > 25)).astype(int)

# Add slight noise to be realistic
for col in Y.columns:
    flip_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    Y.loc[flip_indices, col] = 1 - Y.loc[flip_indices, col]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
model.fit(X_train, Y_train)

# Predict
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Accuracy report
print("🔍 Accuracy Report")
for i, col in enumerate(Y.columns):
    train_acc = accuracy_score(Y_train.iloc[:, i], Y_train_pred[:, i])
    test_acc = accuracy_score(Y_test.iloc[:, i], Y_test_pred[:, i])
    print(f"{col} - Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")
