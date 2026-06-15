import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils import load_dataset, split_features_target, compute_metrics, save_metrics


df = load_dataset("../data/raw/churn.csv")

X, y = split_features_target(df, target_column="churn")

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = compute_metrics(y_test, y_pred)

print("Evaluation metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

save_metrics(metrics, "../artifacts/metrics.txt")

with open("../artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model.pkl")
print("Metrics saved to metrics.txt")
