"""Métricas para clasificación de sentimientos y reputación agregada."""

from __future__ import annotations

import logging
import math
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_LABELS: tuple[str, ...] = ("neg", "neu", "pos")


def _validate_labels(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud")
    known = set(labels)
    unexpected = sorted((set(y_true) | set(y_pred)) - known)
    if unexpected:
        raise ValueError(f"Etiquetas fuera del catálogo esperado: {unexpected}")


def _confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true, pred in zip(y_true, y_pred):
        matrix[index[true]][index[pred]] += 1
    return matrix


def _per_label_stats(matrix: list[list[int]], labels: Sequence[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        stats[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(matrix[i])),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return stats


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
    _validate_labels(y_true, y_pred, labels)

    cm = _confusion_matrix(y_true, y_pred, labels)
    stats = _per_label_stats(cm, labels)
    total = len(y_true)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / total if total else 0.0

    precision_macro = sum(stats[label]["precision"] for label in labels) / len(labels)
    recall_macro = sum(stats[label]["recall"] for label in labels) / len(labels)
    f1_macro = sum(stats[label]["f1"] for label in labels) / len(labels)

    tp_total = sum(stats[label]["tp"] for label in labels)
    fp_total = sum(stats[label]["fp"] for label in labels)
    fn_total = sum(stats[label]["fn"] for label in labels)
    precision_micro = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall_micro = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if precision_micro + recall_micro
        else 0.0
    )

    metrics: dict[str, float | list] = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "confusion_matrix": cm,
        "labels": list(labels),
    }
    logger.info("Evaluación clasificación: acc=%.3f f1_macro=%.3f", metrics["accuracy"], metrics["f1_macro"])
    return metrics


def evaluate_reputation(
    scores_true: dict[str, float],
    scores_pred: dict[str, float],
) -> dict[str, float]:
    """Compara puntuaciones de reputación por aspecto con valores de referencia.

    Calcula MAE, RMSE y correlación de Pearson sobre los aspectos en común.

    Args:
        scores_true: Dict {aspecto: puntuación real 0-5}.
        scores_pred: Dict {aspecto: puntuación predicha 0-5}.

    Returns:
        Diccionario con MAE, RMSE, Pearson y nº de aspectos comparados.
    """
    common = sorted(set(scores_true) & set(scores_pred))
    if not common:
        logger.warning("No hay aspectos en común para evaluar reputación")
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "n_aspects": 0}

    y_true = [float(scores_true[a]) for a in common]
    y_pred = [float(scores_pred[a]) for a in common]

    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(common)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(common))

    mean_true = sum(y_true) / len(y_true)
    mean_pred = sum(y_pred) / len(y_pred)
    var_true = sum((value - mean_true) ** 2 for value in y_true)
    var_pred = sum((value - mean_pred) ** 2 for value in y_pred)
    if len(common) >= 2 and var_true > 0 and var_pred > 0:
        covariance = sum((a - mean_true) * (b - mean_pred) for a, b in zip(y_true, y_pred))
        pearson = covariance / math.sqrt(var_true * var_pred)
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
    _validate_labels(y_true, y_pred, labels)
    matrix = _confusion_matrix(y_true, y_pred, labels)
    stats = _per_label_stats(matrix, labels)
    report: dict[str, dict[str, float]] = {}
    for label in labels:
        report[label] = {
            "precision": float(stats[label]["precision"]),
            "recall": float(stats[label]["recall"]),
            "f1": float(stats[label]["f1"]),
            "support": float(stats[label]["support"]),
        }
    return report


def evaluate_predictions_by_aspect(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    aspects: Sequence[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> dict[str, dict[str, float | list]]:
    """Calcula métricas de clasificación separadas por aspecto.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        aspects: Aspectos paralelos a las etiquetas.
        labels: Etiquetas posibles en orden.

    Returns:
        Dict {aspecto: métricas de `evaluate_predictions`}.
    """
    if not (len(y_true) == len(y_pred) == len(aspects)):
        raise ValueError("y_true, y_pred y aspects deben tener la misma longitud")

    grouped: dict[str, dict[str, list[str]]] = {}
    for true, pred, aspect in zip(y_true, y_pred, aspects):
        bucket = grouped.setdefault(str(aspect), {"y_true": [], "y_pred": []})
        bucket["y_true"].append(true)
        bucket["y_pred"].append(pred)

    return {
        aspect: evaluate_predictions(bucket["y_true"], bucket["y_pred"], labels=labels)
        for aspect, bucket in sorted(grouped.items())
    }
