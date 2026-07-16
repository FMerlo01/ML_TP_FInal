"""Persistencia e inferencia reproducible para el modelo de ingresos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]

MODEL_FEATURE_COLUMNS = [
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
    "capital_gain",
    "capital_loss",
]

NUMERIC_MODEL_COLUMNS = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
]

CATEGORICAL_MODEL_COLUMNS = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
]

TARGET_COLUMN = "income"
DEFAULT_ARTIFACT_PATH = Path("artifacts/random_forest_income.joblib")


def _canonical_column_name(name: Any) -> str:
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "martial_status": "marital_status",  # Typo del encabezado de extra-data.csv.
        "educationnum": "education_num",
        "capitalgain": "capital_gain",
        "capitalloss": "capital_loss",
        "hoursperweek": "hours_per_week",
        "nativecountry": "native_country",
    }
    return aliases.get(normalized, normalized)


def canonicalize_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Normaliza encabezados y espacios sin aplicar transformaciones aprendidas."""
    frame = data.copy()
    frame.columns = [_canonical_column_name(column) for column in frame.columns]

    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Columnas duplicadas luego de normalizar: {duplicated}")

    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].astype("string").str.strip()

    return frame


def _rows_to_frame(rows: Any) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, Mapping):
        return pd.DataFrame([rows])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        values = list(rows)
        if len(values) != len(RAW_FEATURE_COLUMNS):
            raise ValueError(
                "Una fila posicional debe tener 14 valores en el orden de Adult "
                f"sin income; recibió {len(values)}."
            )
        return pd.DataFrame([values], columns=RAW_FEATURE_COLUMNS)
    raise TypeError("rows debe ser DataFrame, diccionario o secuencia de 14 valores.")


def prepare_features(rows: Any) -> pd.DataFrame:
    """Valida una o más filas crudas y devuelve las columnas esperadas por el pipeline."""
    frame = canonicalize_frame(_rows_to_frame(rows))
    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas por el modelo: {missing}")

    features = frame.loc[:, MODEL_FEATURE_COLUMNS].copy()
    for column in NUMERIC_MODEL_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="raise")

    for column in CATEGORICAL_MODEL_COLUMNS:
        features[column] = features[column].astype("string").str.strip()

    features[["workclass", "occupation"]] = features[
        ["workclass", "occupation"]
    ].replace("?", "Sin_dato")

    if features.isna().any().any():
        missing_by_column = features.isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0].to_dict()
        raise ValueError(f"Hay valores faltantes no admitidos: {missing_by_column}")

    return features


def parse_income_target(values: pd.Series) -> pd.Series:
    """Convierte las variantes <=50K[.] y >50K[.] a 0/1."""
    cleaned = values.astype("string").str.strip().str.removesuffix(".")
    target = cleaned.map({"<=50K": 0, ">50K": 1})
    if target.isna().any():
        invalid = sorted(cleaned[target.isna()].dropna().unique().tolist())
        raise ValueError(f"Valores de income no reconocidos: {invalid}")
    return target.astype(int)


def load_labeled_csv(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carga un CSV con encabezado, retorna datos canónicos, features e income binario."""
    raw = canonicalize_frame(pd.read_csv(path, skipinitialspace=True))
    if TARGET_COLUMN not in raw.columns:
        raise ValueError(f"El archivo {path} no contiene la columna {TARGET_COLUMN!r}.")
    return raw, prepare_features(raw), parse_income_target(raw[TARGET_COLUMN])


def save_model_bundle(model: Any, path: str | Path = DEFAULT_ARTIFACT_PATH) -> Path:
    """Guarda el pipeline completo y su contrato de entrada."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("El modelo debe implementar predict_proba.")

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_feature_columns": MODEL_FEATURE_COLUMNS,
            "raw_feature_columns": RAW_FEATURE_COLUMNS,
            "positive_class": ">50K",
        },
        artifact_path,
    )
    return artifact_path


def load_model_bundle(path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    bundle = joblib.load(path)
    required = {"model", "model_feature_columns", "raw_feature_columns", "positive_class"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Artefacto incompleto; faltan claves: {sorted(missing)}")
    if bundle["model_feature_columns"] != MODEL_FEATURE_COLUMNS:
        raise ValueError("El contrato de columnas del artefacto no coincide con el código.")
    return bundle


def predict_rows(
    rows: Any,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """Devuelve probabilidad de >50K y clase para una o más filas nuevas."""
    if not 0 <= decision_threshold <= 1:
        raise ValueError("decision_threshold debe estar entre 0 y 1.")

    features = prepare_features(rows)
    model = load_model_bundle(artifact_path)["model"]
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "probability_above_50k": probabilities,
            "prediction_above_50k": (probabilities >= decision_threshold).astype(int),
        },
        index=features.index,
    )


def predict_one(
    row: Any,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> float:
    """Atajo para obtener solo la probabilidad de una fila."""
    result = predict_rows(row, artifact_path=artifact_path)
    if len(result) != 1:
        raise ValueError("predict_one acepta exactamente una fila.")
    return float(np.asarray(result["probability_above_50k"])[0])
