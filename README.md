# MLOps TP1 — Prédiction du churn client

## Description
Ce projet contient une baseline de classification (RandomForestClassifier) pour prédire
le churn (départ) des clients à partir de caractéristiques telles que le type de contrat,
le mode de paiement, le nombre d'appels au support, etc.

## Prérequis
- Python 3.x
- pip

## Installation
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Exécution
```bash
cd src
python baseline_train.py
```

## Entrées
- `data/raw/churn.csv` : dataset brut contenant les caractéristiques clients et la colonne
  cible `churn` (0 = pas de churn, 1 = churn)

## Sorties
- `artifacts/model.pkl` : modèle RandomForestClassifier entraîné
- `artifacts/metrics.txt` : métriques d'évaluation (accuracy, precision, recall, f1)

## Structure du projet
```
data/raw/        # données brutes
data/processed/  # données post-transformation
src/             # code source (baseline_train.py, utils.py)
artifacts/       # sorties d'entraînement (modèle, métriques)
notebooks/       # exploration
```

## Rôle des outils (Git / MLflow / DVC)
_À compléter à l'étape 12._
