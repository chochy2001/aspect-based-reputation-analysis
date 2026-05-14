"""Construcción de datasets aspect-based a partir de DataFrames de reseñas.

Provee utilidades compartidas por los scripts de entrenamiento y evaluación.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from src.aspects.extractor import extract_aspects, extract_category_mentions, load_spacy_model, map_to_category
from src.classical.lexicon import load_lexicon, score_aspect_lexicon

logger = logging.getLogger(__name__)

DEFAULT_NEUTRAL_BAND: float = 0.1
ASPECT_COLUMN_CANDIDATES: tuple[str, ...] = ("aspect", "aspecto")
LABEL_COLUMN_CANDIDATES: tuple[str, ...] = ("label", "sentiment", "polarity", "polaridad")
LABEL_ALIASES: dict[str, str] = {
    "pos": "pos",
    "positivo": "pos",
    "positive": "pos",
    "neu": "neu",
    "neutro": "neu",
    "neutral": "neu",
    "neg": "neg",
    "negativo": "neg",
    "negative": "neg",
}


def sentiment_label(score: float, neutral_band: float = DEFAULT_NEUTRAL_BAND) -> str:
    """Convierte un score continuo ∈ [-1, 1] a etiqueta categórica.

    Args:
        score: Valor de polaridad.
        neutral_band: Ancho de la banda central considerada neutra.

    Returns:
        Una de las etiquetas 'pos', 'neg' o 'neu'.
    """
    if score > neutral_band:
        return "pos"
    if score < -neutral_band:
        return "neg"
    return "neu"


def _first_existing_column(df: "pd.DataFrame", candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def has_annotation_columns(df: "pd.DataFrame") -> bool:
    """Indica si el DataFrame incluye columnas de aspecto y etiqueta manual."""
    return (
        _first_existing_column(df, ASPECT_COLUMN_CANDIDATES) is not None
        and _first_existing_column(df, LABEL_COLUMN_CANDIDATES) is not None
    )


def normalize_label(label: str) -> str:
    """Normaliza etiquetas comunes a {'pos', 'neg', 'neu'}."""
    normalized = str(label).strip().lower()
    if normalized not in LABEL_ALIASES:
        raise ValueError(f"Etiqueta de sentimiento no válida: {label!r}")
    return LABEL_ALIASES[normalized]


def build_annotated_dataset(
    df: "pd.DataFrame",
    categories_only: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Construye dataset a partir de anotaciones explícitas en el CSV.

    Acepta columnas `aspect`/`aspecto` y `label`/`sentiment`/`polarity`/
    `polaridad`. Las etiquetas se normalizan a `pos`, `neg` o `neu`.
    """
    if "text" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'text'")

    aspect_column = _first_existing_column(df, ASPECT_COLUMN_CANDIDATES)
    label_column = _first_existing_column(df, LABEL_COLUMN_CANDIDATES)
    if aspect_column is None or label_column is None:
        raise ValueError("Faltan columnas de anotación de aspecto/sentimiento")

    texts: list[str] = []
    aspects: list[str] = []
    labels: list[str] = []
    unmapped_aspects: list[tuple[object, str]] = []
    for idx, row in df.iterrows():
        raw_aspect = str(row[aspect_column]).strip()
        if not raw_aspect or raw_aspect.lower() in {"nan", "none"}:
            unmapped_aspects.append((idx, raw_aspect))
            continue
        aspect = map_to_category(raw_aspect) if categories_only else None
        if aspect is None:
            if categories_only:
                unmapped_aspects.append((idx, raw_aspect))
                continue
            aspect = raw_aspect.lower()
        texts.append(str(row["text"]))
        aspects.append(aspect)
        labels.append(normalize_label(str(row[label_column])))
    if unmapped_aspects:
        examples = ", ".join(f"fila {idx}: {aspect!r}" for idx, aspect in unmapped_aspects[:5])
        raise ValueError(
            "Hay anotaciones con aspectos fuera del catálogo esperado "
            f"({len(unmapped_aspects)} casos; ejemplos: {examples})."
        )
    if not texts:
        raise ValueError("No se construyeron ejemplos anotados; revisa columnas de aspecto y etiqueta")
    logger.info("Dataset anotado construido con %d ejemplos", len(texts))
    return texts, aspects, labels


def build_dataset(
    df: "pd.DataFrame",
    categories_only: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Construye triplas (texto, aspecto, label) usando el lexicón como pseudo-label.

    Útil como entrenamiento débilmente supervisado cuando no hay anotaciones
    manuales disponibles.

    Args:
        df: DataFrame con al menos la columna `text`.
        categories_only: Si True, usa las cinco categorías canónicas del proyecto
            (calidad, precio, envío, durabilidad y atención). Si False, conserva
            aspectos crudos no mapeados como respaldo.

    Returns:
        Tres listas paralelas: textos, aspectos y etiquetas.
    """
    if "text" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'text'")

    nlp = load_spacy_model()
    lexicon = load_lexicon()

    texts: list[str] = []
    aspects: list[str] = []
    labels: list[str] = []
    for _, row in df.iterrows():
        text = str(row["text"])
        category_scores: dict[str, list[float]] = {}
        for category, keyword in extract_category_mentions(text):
            category_scores.setdefault(category, []).append(score_aspect_lexicon(text, keyword, lexicon))

        if not categories_only:
            for aspect in extract_aspects(text, nlp):
                category = map_to_category(aspect) or aspect
                if category in category_scores:
                    continue
                category_scores.setdefault(category, []).append(score_aspect_lexicon(text, aspect, lexicon))

        for aspect, scores in category_scores.items():
            score = sum(scores) / len(scores)
            texts.append(text)
            aspects.append(aspect)
            labels.append(sentiment_label(score))
    logger.info("Dataset construido con %d ejemplos", len(texts))
    return texts, aspects, labels


def build_best_available_dataset(
    df: "pd.DataFrame",
    categories_only: bool = True,
) -> tuple[list[str], list[str], list[str], str]:
    """Usa anotaciones reales si existen; si no, genera pseudo-etiquetas.

    Returns:
        Textos, aspectos, etiquetas y una cadena `manual` o `pseudo_lexicon`.
    """
    if has_annotation_columns(df):
        texts, aspects, labels = build_annotated_dataset(df, categories_only=categories_only)
        return texts, aspects, labels, "manual"

    texts, aspects, labels = build_dataset(df, categories_only=categories_only)
    return texts, aspects, labels, "pseudo_lexicon"
