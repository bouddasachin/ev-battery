
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv("battery_cleaned.csv")

# Use capacity and cycle number as initial input features.
# Do NOT include SoH because Health is derived from SoH.
X = df[["Capacity", "Cycle"]]
y = df["Health"]
groups = df["Battery"]

print("Dataset loaded:", len(df), "records")
print("\nClass counts:")
print(y.value_counts())

# Test on a battery excluded from training
logo = LeaveOneGroupOut()
all_actual = []
all_predicted = []

for train_idx, test_idx in logo.split(X, y, groups):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        )
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    all_actual.extend(y_test)
    all_predicted.extend(predictions)

print("\nEvaluation on held-out batteries:")
print(classification_report(
    all_actual,
    all_predicted,
    labels=["Healthy", "Degrading", "Critical"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(
    all_actual,
    all_predicted,
    labels=["Healthy", "Degrading", "Critical"]
))

# Train a final model on all available data
final_model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
)

final_model.fit(X, y)

import joblib
joblib.dump(final_model, "battery_model.pkl")

print("\nFinal model saved as battery_model.pkl")
