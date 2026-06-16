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

| Outil  | Rôle principal | Question à laquelle il répond |
|--------|-----------------|-------------------------------|
| Git    | Versionner le code et les métadonnées du projet | Quelle version du code est utilisée ? |
| MLflow | Suivre les runs, paramètres, métriques et artefacts | Que s'est-il passé pendant cette exécution ? |
| DVC    | Suivre les données ou artefacts hors logique Git | Quelle version de la donnée est associée au projet ? |

## Pipeline

### Structure
Le pipeline est découpé en 4 étapes dans `src/` :
- `prepare.py` : chargement et préparation des données
- `train.py` : entraînement du modèle
- `evaluate.py` : évaluation sur le jeu de test
- `save.py` : sauvegarde des artefacts sur disque

Lancer le pipeline manuellement (depuis la racine, venv activé) :
```bash
python main.py
```

### Orchestration Airflow (WSL uniquement)

#### Installation
```bash
# Dans WSL, depuis la racine du projet
python3.11 -m venv .venv
source .venv/bin/activate
export AIRFLOW_HOME=$(pwd)/airflow_home
pip install "apache-airflow==2.9.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"
airflow db migrate
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
```

#### Lancer Airflow
```bash
# Terminal 1 : scheduler
source .venv/bin/activate && export AIRFLOW_HOME=$(pwd)/airflow_home
airflow scheduler

# Terminal 2 : webserver
source .venv/bin/activate && export AIRFLOW_HOME=$(pwd)/airflow_home
airflow webserver --port 8080
```

#### Déclencher le DAG
Ouvrir http://localhost:8080, se connecter (admin/admin), dépausser `churn_pipeline` et cliquer sur Trigger DAG.

#### Contrôles intégrés
- `check_data` : vérifie que `churn.csv` existe — lève `FileNotFoundError` si absent
- `prepare_data` : vérifie que le dataset n'est pas vide et que la colonne cible existe
- `save_artifacts` : vérifie que les fichiers ont bien été écrits sur disque
- En cas d'échec d'une tâche, toutes les tâches suivantes sont bloquées automatiquement par Airflow