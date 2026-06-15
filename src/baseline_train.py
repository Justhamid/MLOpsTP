import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils import load_dataset, split_features_target, compute_metrics, save_metrics


# Paramètres
n_estimators = 200
max_depth = 5
test_size = 0.2
random_state = 42

df = load_dataset("../data/raw/churn.csv")

X, y = split_features_target(df, target_column="churn")

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=random_state
)

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
        mlflow.log_metric(key, value)

    save_metrics(metrics, "../artifacts/metrics.txt")

    with open("../artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)

    mlflow.log_artifact("../artifacts/metrics.txt")
    mlflow.log_artifact("../artifacts/model.pkl")
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Model saved to artifacts/model.pkl")
    print("Metrics saved to artifacts/metrics.txt")