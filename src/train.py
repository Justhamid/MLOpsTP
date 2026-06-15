from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, n_estimators: int = 100, max_depth: int = 5, random_state: int = 42):
    """
    Entraîne un modèle RandomForestClassifier.

    Entrées :
        X_train (DataFrame) : variables d'entraînement
        y_train (Series) : cible d'entraînement
        n_estimators (int) : nombre d'arbres
        max_depth (int) : profondeur maximale des arbres
        random_state (int) : graine aléatoire

    Sortie :
        model (RandomForestClassifier) : modèle entraîné
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model