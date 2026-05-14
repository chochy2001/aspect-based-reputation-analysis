"""Agrega sentimientos por aspecto a scores de reputación 0-5."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

LABEL_TO_VALUE: dict[str, float] = {"neg": -1.0, "neu": 0.0, "pos": 1.0}
MIN_SCORE: float = 0.0
MAX_SCORE: float = 5.0
CONFIDENCE_K: float = 5.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_sentiment(value: str | float) -> float:
    """Convierte etiqueta o float a [-1, 1].

    Args:
        value: Etiqueta ('pos'/'neg'/'neu') o score numérico.

    Returns:
        Float ∈ [-1, 1].
    """
    if isinstance(value, str):
        label = value.lower()
        if label not in LABEL_TO_VALUE:
            raise ValueError(f"Etiqueta de sentimiento no válida: {value!r}")
        return LABEL_TO_VALUE[label]
    return _clamp(float(value), -1.0, 1.0)


def _normalize_confidence(value: float | int | None) -> float:
    """Normaliza una confianza a [0, 1]."""
    if value is None:
        return 1.0
    return _clamp(float(value), 0.0, 1.0)


def _parse_prediction_value(value: Any) -> tuple[float, float]:
    """Convierte una predicción flexible a (sentimiento, confianza)."""
    if isinstance(value, Mapping):
        sentiment = value.get("sentiment", value.get("label"))
        if sentiment is None:
            raise ValueError("Las predicciones con confianza deben incluir 'sentiment' o 'label'")
        return _normalize_sentiment(sentiment), _normalize_confidence(value.get("confidence"))
    return _normalize_sentiment(value), 1.0


def _sentiment_to_score(sentiment: float) -> float:
    """Mapea [-1, 1] -> [0, 5] linealmente."""
    score = (sentiment + 1.0) * (MAX_SCORE - MIN_SCORE) / 2.0 + MIN_SCORE
    return _clamp(score, MIN_SCORE, MAX_SCORE)


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
    predictions: Iterable[Mapping[str, Any]],
    prior: float = 0.0,
) -> dict[str, float]:
    """Convierte predicciones por aspecto en scores agregados 0-5.

    Cada elemento de `predictions` es un dict {aspecto: sentimiento} para una
    reseña individual. También acepta valores con confianza, por ejemplo
    {"calidad": {"sentiment": "pos", "confidence": 0.82}}.

    Args:
        predictions: Diccionarios {aspecto: predicción}; cada predicción puede
            ser 'pos'|'neg'|'neu', un float en [-1, 1] o un dict con
            `sentiment`/`label` y `confidence`.
        prior: Sentimiento previo ∈ [-1, 1] al que se suaviza cuando hay pocas observaciones.

    Returns:
        Diccionario {aspecto: score ∈ [0, 5]}.
    """
    prior = _normalize_sentiment(prior)
    prediction_list = list(predictions)
    if not prediction_list:
        return {}

    sums: dict[str, float] = defaultdict(float)
    weights: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for pred in prediction_list:
        for aspect, raw_prediction in pred.items():
            sentiment, confidence = _parse_prediction_value(raw_prediction)
            sums[aspect] += sentiment * confidence
            weights[aspect] += confidence
            counts[aspect] += 1

    scores: dict[str, float] = {}
    for aspect, total in sums.items():
        n = counts[aspect]
        avg = total / weights[aspect] if weights[aspect] > 0 else prior
        evidence_weight = _confidence_weight(n)
        adjusted = evidence_weight * avg + (1.0 - evidence_weight) * prior
        scores[aspect] = round(_sentiment_to_score(adjusted), 3)

    logger.info("Calculados scores de reputación para %d aspectos", len(scores))
    return scores


def compute_product_reputation_scores(
    records: Iterable[Mapping[str, Any]],
    prior: float = 0.0,
    product_key: str = "product_id",
) -> dict[str, dict[str, float]]:
    """Agrega predicciones a reputación por producto y aspecto.

    Cada registro debe incluir `product_id`, `aspect`, `sentiment` y, de forma
    opcional, `confidence`.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if product_key not in record:
            raise ValueError(f"Falta la llave requerida {product_key!r}")
        if "aspect" not in record or "sentiment" not in record:
            raise ValueError("Cada registro debe incluir 'aspect' y 'sentiment'")
        product_id = str(record[product_key])
        grouped[product_id].append({
            str(record["aspect"]): {
                "sentiment": record["sentiment"],
                "confidence": record.get("confidence", 1.0),
            }
        })

    return {
        product_id: compute_reputation_scores(predictions, prior=prior)
        for product_id, predictions in grouped.items()
    }


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
        return _clamp(sum(aspect_scores.values()) / len(aspect_scores), MIN_SCORE, MAX_SCORE)

    total_w = 0.0
    weighted = 0.0
    for aspect, score in aspect_scores.items():
        w = weights.get(aspect, 0.0)
        weighted += w * score
        total_w += w
    if total_w == 0.0:
        return _clamp(sum(aspect_scores.values()) / len(aspect_scores), MIN_SCORE, MAX_SCORE)
    return _clamp(weighted / total_w, MIN_SCORE, MAX_SCORE)


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
