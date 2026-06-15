from src.prepare import prepare_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.save import save_artifacts
import mlflow
import mlflow.sklearn


# Paramètres
csv_path = "data/raw/churn.csv"
target_column = "churn"
test_size = 0.2
random_state = 42
n_estimators = 200
max_depth = 5
output_dir = "artifacts/"


with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    # Étape 1 : préparation
    X_train, X_test, y_train, y_test = prepare_data(
        csv_path=csv_path,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state
    )

    # Étape 2 : entraînement
    model = train_model(
        X_train, y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    # Étape 3 : évaluation
    metrics = evaluate_model(model, X_test, y_test)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
        mlflow.log_metric(key, value)

    # Étape 4 : sauvegarde
    save_artifacts(model, metrics, output_dir=output_dir)

    mlflow.log_artifact(f"{output_dir}/metrics.txt")
    mlflow.log_artifact(f"{output_dir}/model.pkl")
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Model saved to {output_dir}model.pkl")
    print(f"Metrics saved to {output_dir}metrics.txt")