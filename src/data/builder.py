"""Construcción de datasets aspect-based a partir de DataFrames de reseñas.

Provee utilidades compartidas por los scripts de entrenamiento y evaluación.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from src.aspects.extractor import extract_aspects, load_spacy_model
from src.classical.lexicon import load_lexicon, score_aspect_lexicon

logger = logging.getLogger(__name__)

DEFAULT_NEUTRAL_BAND: float = 0.1


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


def build_dataset(df: "pd.DataFrame") -> tuple[list[str], list[str], list[str]]:
    """Construye triplas (texto, aspecto, label) usando el lexicón como pseudo-label.

    Útil como entrenamiento débilmente supervisado cuando no hay anotaciones
    manuales disponibles.

    Args:
        df: DataFrame con al menos la columna `text`.

    Returns:
        Tres listas paralelas: textos, aspectos y etiquetas.
    """
    nlp = load_spacy_model()
    lexicon = load_lexicon()

    texts: list[str] = []
    aspects: list[str] = []
    labels: list[str] = []
    for _, row in df.iterrows():
        text = str(row["text"])
        for aspect in extract_aspects(text, nlp):
            score = score_aspect_lexicon(text, aspect, lexicon)
            texts.append(text)
            aspects.append(aspect)
            labels.append(sentiment_label(score))
    logger.info("Dataset construido con %d ejemplos", len(texts))
    return texts, aspects, labels
