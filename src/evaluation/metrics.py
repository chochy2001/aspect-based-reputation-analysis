"""Métricas para clasificación de sentimientos y reputación agregada."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

DEFAULT_LABELS: tuple[str, ...] = ("neg", "neu", "pos")


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> dict[str, float | list]:
    """Calcula accuracy, precision, recall, F1 (macro/micro) y matriz de confusión.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        labels: Etiquetas posibles en orden.

    Returns:
        Diccionario con métricas. La matriz de confusión se devuelve como lista
        de listas para facilitar serialización.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud")

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    metrics: dict[str, float | list] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
    }
    logger.info("Evaluación clasificación: acc=%.3f f1_macro=%.3f", metrics["accuracy"], metrics["f1_macro"])
    return metrics


def evaluate_reputation(
    scores_true: dict[str, float],
    scores_pred: dict[str, float],
) -> dict[str, float]:
    """Compara scores de reputación por aspecto con valores de referencia.

    Calcula MAE, RMSE y correlación de Pearson sobre los aspectos en común.

    Args:
        scores_true: Dict {aspecto: score real 0-5}.
        scores_pred: Dict {aspecto: score predicho 0-5}.

    Returns:
        Diccionario con MAE, RMSE, Pearson y nº de aspectos comparados.
    """
    common = sorted(set(scores_true) & set(scores_pred))
    if not common:
        logger.warning("No hay aspectos en común para evaluar reputación")
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "n_aspects": 0}

    y_true = np.array([scores_true[a] for a in common], dtype=float)
    y_pred = np.array([scores_pred[a] for a in common], dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    if len(common) >= 2 and y_true.std() > 0 and y_pred.std() > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "n_aspects": len(common),
    }
    logger.info("Evaluación reputación: MAE=%.3f RMSE=%.3f Pearson=%.3f", mae, rmse, pearson)
    return metrics


def classification_report_dict(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> dict[str, dict[str, float]]:
    """Devuelve precision/recall/F1 por clase.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        labels: Etiquetas posibles.

    Returns:
        Dict {label: {precision, recall, f1, support}}.
    """
    report: dict[str, dict[str, float]] = {}
    for label in labels:
        report[label] = {
            "precision": float(precision_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)),
            "support": float(sum(1 for y in y_true if y == label)),
        }
    return report
