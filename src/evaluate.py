from .utils import compute_metrics


def evaluate_model(model, X_test, y_test):
    """
    Évalue un modèle entraîné sur un jeu de test.

    Entrées :
        model (RandomForestClassifier) : modèle entraîné
        X_test (DataFrame) : variables de test
        y_test (Series) : cible de test

    Sortie :
        metrics (dict) : dictionnaire contenant accuracy, precision, recall, f1
    """
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    return metrics