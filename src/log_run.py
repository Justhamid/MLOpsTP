import mlflow
import mlflow.sklearn


def log_run(params: dict, metrics: dict, model, artifacts_dir: str):
    """
    Trace un run MLflow complet.

    Entrées :
        params (dict) : paramètres du run (hyperparamètres, config)
        metrics (dict) : métriques d'évaluation
        model : modèle entraîné
        artifacts_dir (str) : dossier contenant les artefacts à logger

    Sortie :
        run_id (str) : identifiant unique du run MLflow
    """
    with mlflow.start_run() as run:

        for key, value in params.items():
            mlflow.log_param(key, value)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_artifact(f"{artifacts_dir}/metrics.txt")
        mlflow.log_artifact(f"{artifacts_dir}/model.pkl")
        mlflow.sklearn.log_model(model, artifact_path="model")

        return run.info.run_id