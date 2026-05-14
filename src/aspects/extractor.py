"""Extracción de aspectos a partir de reseñas usando reglas y dependencias de spaCy."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy
    from spacy.language import Language

logger = logging.getLogger(__name__)

ASPECT_POS_TAGS: frozenset[str] = frozenset({"NOUN", "PROPN"})
MIN_ASPECT_LENGTH: int = 3

ASPECT_BLOCKLIST: frozenset[str] = frozenset({
    "cosa", "cosas", "vez", "veces", "tipo", "forma", "manera",
    "producto", "artículo", "compra",
})

ASPECT_CATEGORIES: dict[str, list[str]] = {
    "calidad": [
        "calidad", "material", "acabado", "fabricación", "pantalla", "display",
        "resolución", "sonido", "audio", "altavoz", "auriculares", "edición",
        "impresión", "tela", "corte", "diseño", "antiadherente",
    ],
    "precio": ["precio", "coste", "costo", "valor"],
    "envío": ["envío", "envio", "entrega", "paquete", "embalaje"],
    "durabilidad": [
        "durabilidad", "batería", "bateria", "pila", "duración", "duracion",
        "autonomía", "autonomia", "resistente", "resistencia", "lavados",
        "desgastó", "desgasto", "roto", "rota", "frágil", "fragil",
    ],
    "atención": ["servicio", "atención", "atencion", "soporte", "vendedor", "cliente", "reclamación"],
}


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_for_match(text: str) -> str:
    return _strip_accents(text.lower())


def _contains_keyword(text: str, keyword: str) -> bool:
    pattern = rf"(?<!\w){re.escape(_normalize_for_match(keyword))}(?!\w)"
    return re.search(pattern, _normalize_for_match(text)) is not None


def extract_category_mentions(text: str) -> list[tuple[str, str]]:
    """Devuelve menciones detectadas como pares (categoría, keyword).

    La detección por palabras clave mantiene el flujo funcionando incluso
    cuando no hay modelo estadístico de spaCy instalado.
    """
    if not text or not isinstance(text, str):
        return []

    mentions: list[tuple[str, str]] = []
    seen: set[str] = set()
    for category, keywords in ASPECT_CATEGORIES.items():
        for keyword in keywords:
            if _contains_keyword(text, keyword):
                key = f"{category}:{_normalize_for_match(keyword)}"
                if key not in seen:
                    mentions.append((category, keyword))
                    seen.add(key)
    return mentions


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
    try:
        import spacy
    except ImportError:
        logger.warning("spaCy no está instalado; usando extracción por palabras clave")
        return None

    try:
        logger.info("Cargando modelo spaCy '%s'", preferred)
        return spacy.load(preferred)
    except OSError:
        logger.warning("Modelo '%s' no disponible; usando fallback '%s'", preferred, fallback)
        try:
            return spacy.load(fallback)
        except OSError:
            logger.warning("Modelo '%s' no disponible; usando extracción por palabras clave", fallback)
            return None


def extract_aspects(text: str, nlp: "Language | None" = None) -> list[str]:
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

    aspects: list[str] = []
    seen: set[str] = set()

    for _, keyword in extract_category_mentions(text):
        key = _normalize_for_match(keyword)
        if key not in seen:
            seen.add(key)
            aspects.append(keyword.lower())

    if nlp is None:
        return aspects

    doc = nlp(text)
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
    aspect_low = _normalize_for_match(aspect)
    for category, keywords in ASPECT_CATEGORIES.items():
        if aspect_low in keywords:
            return category
        if aspect_low in {_normalize_for_match(keyword) for keyword in keywords}:
            return category
    return None


def extract_aspect_phrases(text: str, nlp: "Language | None") -> list[tuple[str, str]]:
    """Extrae pares (aspecto, modificador) usando dependencias sintácticas.

    Args:
        text: Texto de la reseña.
        nlp: Modelo spaCy cargado.

    Returns:
        Lista de tuplas (sustantivo, adjetivo) relacionadas por dependencia.
    """
    if nlp is None:
        return []

    doc = nlp(text)
    pairs: list[tuple[str, str]] = []
    for token in doc:
        if token.pos_ == "ADJ" and token.head.pos_ in ASPECT_POS_TAGS:
            pairs.append((token.head.lemma_.lower(), token.lemma_.lower()))
    return pairs
