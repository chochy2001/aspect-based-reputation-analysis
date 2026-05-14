"""Extracción de aspectos a partir de reseñas usando reglas y dependencias de spaCy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy
    from spacy.language import Language

logger = logging.getLogger(__name__)

ASPECT_POS_TAGS: frozenset[str] = frozenset({"NOUN", "PROPN"})
MIN_ASPECT_LENGTH: int = 3

ASPECT_BLOCKLIST: frozenset[str] = frozenset({
    "cosa", "cosas", "vez", "veces", "tipo", "forma", "manera",
    "producto", "artículo", "compra", "envío",
})

ASPECT_CATEGORIES: dict[str, list[str]] = {
    "calidad": ["calidad", "material", "acabado", "fabricación"],
    "precio": ["precio", "coste", "costo", "valor"],
    "envio": ["envío", "entrega", "paquete", "embalaje"],
    "bateria": ["batería", "pila", "duración", "autonomía"],
    "pantalla": ["pantalla", "display", "resolución"],
    "sonido": ["sonido", "audio", "altavoz", "auriculares"],
    "servicio": ["servicio", "atención", "soporte", "vendedor"],
}


def load_spacy_model(preferred: str = "es_core_news_lg", fallback: str = "es_core_news_sm"):
    """Carga un modelo de spaCy en español con fallback.

    Args:
        preferred: Modelo preferido.
        fallback: Modelo a usar si el preferido no está disponible.

    Returns:
        Modelo spaCy cargado.

    Raises:
        OSError: Si ningún modelo está disponible.
    """
    import spacy

    try:
        logger.info("Cargando modelo spaCy '%s'", preferred)
        return spacy.load(preferred)
    except OSError:
        logger.warning("Modelo '%s' no disponible; usando fallback '%s'", preferred, fallback)
        return spacy.load(fallback)


def extract_aspects(text: str, nlp: "Language") -> list[str]:
    """Extrae aspectos relevantes de un texto usando POS tagging.

    Filtra sustantivos y nombres propios significativos eliminando ruido común.

    Args:
        text: Texto de la reseña.
        nlp: Modelo spaCy cargado.

    Returns:
        Lista de aspectos únicos (lemmas en minúsculas).
    """
    if not text or not isinstance(text, str):
        return []

    doc = nlp(text)
    aspects: list[str] = []
    seen: set[str] = set()

    for token in doc:
        if token.pos_ not in ASPECT_POS_TAGS:
            continue
        lemma = token.lemma_.lower().strip()
        if len(lemma) < MIN_ASPECT_LENGTH:
            continue
        if lemma in ASPECT_BLOCKLIST:
            continue
        if lemma in seen:
            continue
        seen.add(lemma)
        aspects.append(lemma)

    return aspects


def map_to_category(aspect: str) -> str | None:
    """Mapea un aspecto crudo a una categoría predefinida.

    Args:
        aspect: Aspecto extraído.

    Returns:
        Nombre de la categoría o None si no encaja.
    """
    aspect_low = aspect.lower()
    for category, keywords in ASPECT_CATEGORIES.items():
        if aspect_low in keywords:
            return category
    return None


def extract_aspect_phrases(text: str, nlp: "Language") -> list[tuple[str, str]]:
    """Extrae pares (aspecto, modificador) usando dependencias sintácticas.

    Args:
        text: Texto de la reseña.
        nlp: Modelo spaCy cargado.

    Returns:
        Lista de tuplas (sustantivo, adjetivo) relacionadas por dependencia.
    """
    doc = nlp(text)
    pairs: list[tuple[str, str]] = []
    for token in doc:
        if token.pos_ == "ADJ" and token.head.pos_ in ASPECT_POS_TAGS:
            pairs.append((token.head.lemma_.lower(), token.lemma_.lower()))
    return pairs
