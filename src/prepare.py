import os
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import load_dataset, split_features_target


def prepare_data(csv_path: str, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Prépare les données pour l'entraînement.

    Entrées :
        csv_path (str) : chemin vers le fichier CSV brut
        target_column (str) : nom de la colonne cible
        test_size (float) : proportion des données réservées au test
        random_state (int) : graine aléatoire pour la reproductibilité

    Sortie :
        tuple (X_train, X_test, y_train, y_test)

    Lève :
        FileNotFoundError : si csv_path n'existe pas
        ValueError : si le dataset est vide
        KeyError : si target_column n'est pas une colonne du dataset
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset introuvable : {csv_path}")

    df = load_dataset(csv_path)

    if df.empty:
        raise ValueError("Le dataset est vide.")

    if target_column not in df.columns:
        raise KeyError(f"Colonne cible '{target_column}' absente du dataset.")

    X, y = split_features_target(df, target_column=target_column)

    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )