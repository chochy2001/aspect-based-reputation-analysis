"""Agrega sentimientos por aspecto a scores de reputación 0-5."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Mapping

logger = logging.getLogger(__name__)

LABEL_TO_VALUE: dict[str, float] = {"neg": -1.0, "neu": 0.0, "pos": 1.0}
MIN_SCORE: float = 0.0
MAX_SCORE: float = 5.0
CONFIDENCE_K: float = 5.0


def _normalize_sentiment(value: str | float) -> float:
    """Convierte etiqueta o float a [-1, 1].

    Args:
        value: Etiqueta ('pos'/'neg'/'neu') o score numérico.

    Returns:
        Float ∈ [-1, 1].
    """
    if isinstance(value, str):
        return LABEL_TO_VALUE.get(value.lower(), 0.0)
    return max(-1.0, min(1.0, float(value)))


def _sentiment_to_score(sentiment: float) -> float:
    """Mapea [-1, 1] -> [0, 5] linealmente."""
    return (sentiment + 1.0) * (MAX_SCORE - MIN_SCORE) / 2.0 + MIN_SCORE


def _confidence_weight(n: int, k: float = CONFIDENCE_K) -> float:
    """Peso de confianza tipo Bayesiano: n / (n + k).

    Args:
        n: Número de observaciones.
        k: Constante de suavizado.

    Returns:
        Valor en [0, 1).
    """
    return n / (n + k) if n > 0 else 0.0


def compute_reputation_scores(
    predictions: list[Mapping[str, str | float]],
    prior: float = 0.0,
) -> dict[str, float]:
    """Convierte predicciones por aspecto en scores agregados 0-5.

    Cada elemento de `predictions` es un dict {aspecto: sentimiento} para una
    reseña individual. La función agrupa por aspecto, promedia y aplica
    un ajuste de confianza hacia un prior (sentimiento neutro por defecto).

    Args:
        predictions: Lista de diccionarios {aspecto: 'pos'|'neg'|'neu'|float}.
        prior: Sentimiento previo ∈ [-1, 1] al que se suaviza cuando hay pocas observaciones.

    Returns:
        Diccionario {aspecto: score ∈ [0, 5]}.
    """
    if not predictions:
        return {}

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for pred in predictions:
        for aspect, sentiment in pred.items():
            sums[aspect] += _normalize_sentiment(sentiment)
            counts[aspect] += 1

    scores: dict[str, float] = {}
    for aspect, total in sums.items():
        n = counts[aspect]
        avg = total / n
        weight = _confidence_weight(n)
        adjusted = weight * avg + (1.0 - weight) * prior
        scores[aspect] = round(_sentiment_to_score(adjusted), 3)

    logger.info("Calculados scores de reputación para %d aspectos", len(scores))
    return scores


def compute_global_reputation(
    aspect_scores: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Calcula un score global a partir de scores por aspecto.

    Args:
        aspect_scores: Dict {aspecto: score 0-5}.
        weights: Pesos opcionales por aspecto; si es None se ponderan por igual.

    Returns:
        Score global ∈ [0, 5].
    """
    if not aspect_scores:
        return 0.0
    if weights is None:
        return sum(aspect_scores.values()) / len(aspect_scores)

    total_w = 0.0
    weighted = 0.0
    for aspect, score in aspect_scores.items():
        w = weights.get(aspect, 0.0)
        weighted += w * score
        total_w += w
    if total_w == 0.0:
        return sum(aspect_scores.values()) / len(aspect_scores)
    return weighted / total_w


def aspect_score_summary(scores: Mapping[str, float]) -> dict[str, str]:
    """Genera etiquetas cualitativas a partir de scores 0-5.

    Args:
        scores: Dict {aspecto: score}.

    Returns:
        Dict {aspecto: etiqueta cualitativa}.
    """
    summary: dict[str, str] = {}
    for aspect, score in scores.items():
        if score >= 4.0:
            summary[aspect] = "muy positivo"
        elif score >= 3.0:
            summary[aspect] = "positivo"
        elif score >= 2.0:
            summary[aspect] = "neutro"
        elif score >= 1.0:
            summary[aspect] = "negativo"
        else:
            summary[aspect] = "muy negativo"
    return summary
