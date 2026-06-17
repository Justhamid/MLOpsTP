import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score

ACCURACY_THRESHOLD = 0.75


def test_model_accuracy_above_threshold():
    """
    Test de non-régression : l'accuracy du modèle sur le jeu de validation
    fixe doit rester >= 0.75.
    Si ce test échoue, une régression de performance a été introduite.
    """
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)

    df_val = pd.read_csv("data/processed/validation.csv")
    X_val = df_val.drop(columns=["churn"])
    y_val = df_val["churn"]

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    assert accuracy >= ACCURACY_THRESHOLD, (
        f"Régression détectée : accuracy={accuracy:.3f} < seuil={ACCURACY_THRESHOLD}"
    )


def test_feature_columns_unchanged():
    """
    Test de non-régression : les colonnes du modèle ne doivent pas changer.
    Si ce test échoue, le schéma des données a été modifié.
    """
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("artifacts/feature_columns.json", "r") as f:
        expected_columns = json.load(f)

    actual_columns = model.feature_names_in_.tolist()

    assert actual_columns == expected_columns, (
        f"Colonnes modifiées !\n"
        f"Attendu : {expected_columns}\n"
        f"Obtenu  : {actual_columns}"
    )


def test_model_returns_binary_predictions():
    """
    Test de non-régression : le modèle ne doit retourner que 0 ou 1.
    """
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)

    df_val = pd.read_csv("data/processed/validation.csv")
    X_val = df_val.drop(columns=["churn"])

    y_pred = model.predict(X_val)

    assert set(y_pred).issubset({0, 1}), (
        f"Le modèle retourne des valeurs inattendues : {set(y_pred)}"
    )