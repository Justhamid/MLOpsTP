import os
import sys
import pickle
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, "/home/a945226/projects/MLOpsTP")

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.prepare import prepare_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.save import save_artifacts

CSV_PATH = "/home/a945226/projects/MLOpsTP/data/raw/churn.csv"
PROCESSED_DIR = "/home/a945226/projects/MLOpsTP/data/processed"
ARTIFACTS_DIR = "/home/a945226/projects/MLOpsTP/artifacts"


def task_check_data(**context):
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset introuvable : {CSV_PATH}")
    print(f"Dataset trouvé : {CSV_PATH}")
    return CSV_PATH


def task_prepare_data(**context):
    csv_path = context["ti"].xcom_pull(task_ids="check_data")

    X_train, X_test, y_train, y_test = prepare_data(
        csv_path=csv_path,
        target_column="churn"
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

    print(f"Données préparées : {X_train.shape[0]} lignes entraînement, {X_test.shape[0]} lignes test")
    return PROCESSED_DIR


def task_train_model(**context):
    processed_dir = context["ti"].xcom_pull(task_ids="prepare_data")

    X_train = pd.read_csv(f"{processed_dir}/X_train.csv")
    y_train = pd.read_csv(f"{processed_dir}/y_train.csv").squeeze()

    model = train_model(
        X_train, y_train,
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_path = f"{ARTIFACTS_DIR}/model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modèle entraîné et sauvegardé : {model_path}")
    return model_path


def task_evaluate_model(**context):
    processed_dir = context["ti"].xcom_pull(task_ids="prepare_data")
    model_path = context["ti"].xcom_pull(task_ids="train_model")

    X_test = pd.read_csv(f"{processed_dir}/X_test.csv")
    y_test = pd.read_csv(f"{processed_dir}/y_test.csv").squeeze()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    metrics = evaluate_model(model, X_test, y_test)

    print("Métriques :")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    context["ti"].xcom_push(key="metrics", value=metrics)
    context["ti"].xcom_push(key="model_path", value=model_path)

    return metrics


def task_save_artifacts(**context):
    metrics = context["ti"].xcom_pull(task_ids="evaluate_model", key="metrics")
    model_path = context["ti"].xcom_pull(task_ids="evaluate_model", key="model_path")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    save_artifacts(model, metrics, output_dir=ARTIFACTS_DIR)

    print(f"Artefacts sauvegardés dans {ARTIFACTS_DIR}")


default_args = {
    "owner": "etudiant",
    "retry_delay": timedelta(minutes=1)
    # plus de retries global ici — chaque tâche a le sien
}

with DAG(
    dag_id="churn_pipeline",
    default_args=default_args,
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "tp-jour2"]
) as dag:

    check = PythonOperator(
        task_id="check_data",
        python_callable=task_check_data,
        retries=3,                          # 3 retries : fichier potentiellement absent temporairement
        retry_delay=timedelta(seconds=30)   # attente courte entre chaque retry
    )

    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=task_prepare_data,
        retries=2,                          # 2 retries : erreur possible sur lecture/écriture disque
        retry_delay=timedelta(seconds=30)
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
        retries=1,                          # 1 retry : entraînement long, on réessaie une fois
        retry_delay=timedelta(minutes=1)
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
        retries=1,
        retry_delay=timedelta(minutes=1)
    )

    save = PythonOperator(
        task_id="save_artifacts",
        python_callable=task_save_artifacts,
        retries=2,                          # 2 retries : écriture disque peut échouer temporairement
        retry_delay=timedelta(seconds=30)
    )

    check >> prepare >> train >> evaluate >> save
