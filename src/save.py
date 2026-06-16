import os
import pickle
import json

from .utils import save_metrics


def save_artifacts(model, metrics, output_dir: str):
    """
    Sauvegarde le modèle entraîné et les métriques sur disque.

    Entrées :
        model (RandomForestClassifier) : modèle entraîné
        metrics (dict) : métriques d'évaluation
        output_dir (str) : dossier de destination

    Sortie :
        aucune valeur retournée — effet de bord : écrit model.pkl et metrics.txt dans output_dir

    Lève :
        RuntimeError : si model.pkl ou metrics.txt n'a pas été correctement créé
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.pkl")
    metrics_path = os.path.join(output_dir, "metrics.txt")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Sauvegarde feature_columns.json
    if hasattr(model, 'feature_names_in_'):
        feature_columns = model.feature_names_in_.tolist()
        with open(os.path.join(output_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
            json.dump(feature_columns, f, indent=2)
    save_metrics(metrics, metrics_path)

    if not os.path.exists(model_path):
        raise RuntimeError(f"Le modèle n'a pas été sauvegardé : {model_path}")

    if not os.path.exists(metrics_path):
        raise RuntimeError(f"Les métriques n'ont pas été sauvegardées : {metrics_path}")