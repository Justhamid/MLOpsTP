from src.prepare import prepare_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.save import save_artifacts
from src.log_run import log_run


# Paramètres centralisés
params = {
    "model_type": "RandomForestClassifier",
    "n_estimators": 200,
    "max_depth": 5,
    "test_size": 0.2,
    "random_state": 42
}

# Étape 1 : préparation
X_train, X_test, y_train, y_test = prepare_data(
    csv_path="data/raw/churn.csv",
    target_column="churn",
    test_size=params["test_size"],
    random_state=params["random_state"]
)

# Étape 2 : entraînement
model = train_model(
    X_train, y_train,
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"]
)

# Étape 3 : évaluation
metrics = evaluate_model(model, X_test, y_test)

print("Evaluation metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# Étape 4 : sauvegarde
save_artifacts(model, metrics, output_dir="artifacts/")

# Étape 5 : traçabilité MLflow
run_id = log_run(
    params=params,
    metrics=metrics,
    model=model,
    artifacts_dir="artifacts/"
)

print(f"Run MLflow enregistré : {run_id}")