import pytest
import pandas as pd
from src.prepare import prepare_data


def test_prepare_data_returns_four_objects(tmp_path):
    """prepare_data doit retourner un tuple de 4 éléments."""
    df = pd.DataFrame({
        "age": [25, 35, 45, 55, 65, 30],
        "tenure_months": [1, 12, 24, 36, 48, 60],
        "monthly_charges": [20.0, 50.0, 70.0, 80.0, 90.0, 100.0],
        "support_calls": [0, 1, 2, 3, 4, 5],
        "contract_type": ["Monthly", "One year", "Two year", "Monthly", "One year", "Two year"],
        "payment_method": ["Electronic check"] * 6,
        "internet_service": ["Fiber"] * 6,
        "churn": [0, 1, 0, 1, 0, 1]
    })
    csv_path = tmp_path / "mini.csv"
    df.to_csv(csv_path, index=False)
    result = prepare_data(str(csv_path), target_column="churn")
    assert len(result) == 4


def test_prepare_data_raises_on_missing_file():
    """prepare_data doit lever FileNotFoundError si le fichier est absent."""
    with pytest.raises(FileNotFoundError):
        prepare_data("fichier_inexistant.csv", target_column="churn")


def test_prepare_data_raises_on_missing_target(tmp_path):
    """prepare_data doit lever KeyError si la colonne cible est absente."""
    df = pd.DataFrame({
        "age": [25, 35],
        "monthly_charges": [20.0, 30.0]
    })
    csv_path = tmp_path / "no_target.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(KeyError):
        prepare_data(str(csv_path), target_column="churn")


def test_prepare_data_raises_on_empty_dataset(tmp_path):
    """prepare_data doit lever ValueError si le dataset est vide."""
    df = pd.DataFrame(columns=["age", "tenure_months", "churn"])
    csv_path = tmp_path / "empty.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        prepare_data(str(csv_path), target_column="churn")


def test_prepare_data_split_sizes(tmp_path):
    """X_train doit être plus grand que X_test (test_size=0.2)."""
    df = pd.DataFrame({
        "age": [25, 35, 45, 55, 65, 30],
        "tenure_months": [1, 12, 24, 36, 48, 60],
        "monthly_charges": [20.0, 50.0, 70.0, 80.0, 90.0, 100.0],
        "support_calls": [0, 1, 2, 3, 4, 5],
        "contract_type": ["Monthly", "One year", "Two year", "Monthly", "One year", "Two year"],
        "payment_method": ["Electronic check"] * 6,
        "internet_service": ["Fiber"] * 6,
        "churn": [0, 1, 0, 1, 0, 1]
    })
    csv_path = tmp_path / "mini.csv"
    df.to_csv(csv_path, index=False)
    X_train, X_test, y_train, y_test = prepare_data(str(csv_path), target_column="churn")
    assert len(X_train) > len(X_test)