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

## Service d'inférence

### Lancer l'API
```bash
uvicorn app:app --reload --port 8000
```

L'interface Swagger est disponible sur : http://localhost:8000/docs

### Format d'entrée

```json
{
  "age": 35,
  "tenure_months": 12,
  "monthly_charges": 75.5,
  "support_calls": 2,
  "contract_type": "Monthly",
  "payment_method": "Electronic check",
  "internet_service": "Fiber"
}
```

### Exemple curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "tenure_months": 12, "monthly_charges": 75.5, "support_calls": 2, "contract_type": "Monthly", "payment_method": "Electronic check", "internet_service": "Fiber"}'
```

### Routes disponibles

| Route | Méthode | Description |
|---|---|---|
| `/` | GET | Statut de l'API |
| `/health` | GET | Santé du service et modèle chargé |
| `/predict` | POST | Prédiction de churn pour un client |
| `/metrics` | GET | Indicateurs de supervision |
| `/drift_report` | POST | Génère un rapport de dérive Evidently |

---

## Supervision

### Indicateurs exposés par `/metrics`

| Indicateur | Description |
|---|---|
| `n_requests` | Nombre total de requêtes reçues |
| `n_errors` | Nombre d'erreurs internes (HTTP 500) |
| `n_invalid_inputs` | Nombre d'entrées hors plage d'entraînement |
| `drift_alerts` | Nombre d'alertes de dérive détectées |
| `predictions_distribution` | Répartition churn / no_churn |
| `evidently_reports_generated` | Nombre de rapports Evidently générés |

### Comportement en cas d'anomalie

Une **anomalie** est une valeur valide en type mais hors de la plage observée à l'entraînement.

Exemple : `age=90` est un entier valide, mais le modèle n'a été entraîné que sur des âges entre 18 et 80.

Comportement : HTTP 200 avec `prediction=null` et un champ `warning` décrivant l'anomalie. La prédiction est bloquée, `n_invalid_inputs` est incrémenté.

```json
{
  "prediction": null,
  "warning": "anomalie détectée",
  "details": ["age=90 hors plage [18, 80]"]
}
```

### Comportement en cas de dérive

Une **dérive** est un déplacement progressif de la distribution des données sur les 50 dernières requêtes.

Comportement : la prédiction est retournée normalement (HTTP 200), mais un champ `drift_alert` est ajouté à la réponse et un WARNING est écrit dans les logs. `drift_alerts` est incrémenté.

```json
{
  "prediction": 0,
  "label": "no_churn",
  "confidence": 0.69,
  "drift_alert": {"tenure_months": 1.53}
}
```

### Emplacement des logs

Les logs sont écrits dans `logs/api.log` avec 3 niveaux :
- `INFO` : flux normal (requête reçue, prédiction effectuée)
- `WARNING` : anomalie ou dérive détectée
- `ERROR` : exception avec stack trace complète

---

## Rapports de dérive

### Déclencher un rapport Evidently

```bash
curl -X POST http://localhost:8000/drift_report
```

Le rapport nécessite **20 requêtes minimum** dans le buffer.

### Emplacement des rapports

Les rapports HTML sont générés dans `logs/drift_reports/drift_YYYYMMDD_HHMMSS.html`.

Ouvre le fichier directement dans ton navigateur (double-clic depuis l'explorateur).

### Interpréter un rapport

Le rapport Evidently compare la distribution des données d'entraînement (référence) avec les 20 dernières requêtes reçues. Il indique pour chaque variable :
- si une dérive est détectée (test statistique de Kolmogorov-Smirnov pour les numériques)
- la p-value et le score de dérive
- des visualisations de distribution côte à côte

Un score de dérive > 0.5 sur une variable signale que sa distribution actuelle s'éloigne significativement de ce que le modèle a vu à l'entraînement.

## CI/CD

### Intégration continue (CI)

Les vérifications automatisées tournent avec **GitHub Actions** à chaque push sur `main`.

**Workflow `.github/workflows/ci.yml` :**
- Installe Python 3.11 et les dépendances
- Réentraîne le modèle depuis les données
- Vérifie que les artefacts sont produits
- Lance les 18 tests pytest (pipeline, API, non-régression)
- Vérifie que l'API démarre et répond sur `/health`

**Lancer les tests en local :**
```bash
pytest tests/ -v
```

**Le workflow se déclenche :**
- À chaque push sur `main` ou `develop`
- À chaque pull request vers `main`

### Ce qui relèverait du déploiement continu (CD)

Le CD n'est pas encore implémenté mais voici ce qui serait ajouté :

- **Publier l'image Docker** de l'API sur un registry (Docker Hub, ECR) après validation des tests
- **Déployer l'API** sur un environnement de staging automatiquement après un run vert
- **Enregistrer le modèle** dans le MLflow Model Registry avec le tag `staging` après réentraînement validé
- **Déclencher un redéploiement** automatique en production après promotion manuelle depuis staging