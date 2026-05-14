"""Preprocesamiento de texto en español: limpieza, tokenización y normalización."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
MULTI_SPACE_PATTERN = re.compile(r"\s+")
NON_ALPHA_PATTERN = re.compile(r"[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ\s]")

SPANISH_STOPWORDS: frozenset[str] = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "a", "en", "por", "para", "con", "sin",
    "y", "o", "u", "pero", "ni", "que", "como", "es", "son",
    "fue", "fueron", "ser", "estar", "este", "esta", "estos", "estas",
    "lo", "le", "les", "me", "te", "se", "su", "sus", "mi", "tu",
})


def strip_accents(text: str) -> str:
    """Elimina acentos manteniendo la ñ.

    Args:
        text: Cadena de entrada.

    Returns:
        Cadena sin acentos.
    """
    text = text.replace("ñ", "\x00").replace("Ñ", "\x01")
    nfkd = unicodedata.normalize("NFKD", text)
    cleaned = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return cleaned.replace("\x00", "ñ").replace("\x01", "Ñ")


def clean_text(text: str, lowercase: bool = True, remove_accents: bool = False) -> str:
    """Limpia un texto eliminando URLs, menciones y caracteres no alfabéticos.

    Args:
        text: Texto crudo.
        lowercase: Convertir a minúsculas.
        remove_accents: Eliminar acentos.

    Returns:
        Texto limpio.
    """
    if not isinstance(text, str):
        return ""
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    if lowercase:
        text = text.lower()
    if remove_accents:
        text = strip_accents(text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """Tokeniza un texto en español por espacios.

    Args:
        text: Texto previamente limpio.
        remove_stopwords: Si True, descarta stopwords en español.

    Returns:
        Lista de tokens.
    """
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in SPANISH_STOPWORDS]
    return tokens


def preprocess_corpus(
    texts: Iterable[str],
    remove_stopwords: bool = False,
    remove_accents: bool = False,
) -> list[list[str]]:
    """Aplica clean + tokenize a un corpus.

    Args:
        texts: Iterable de strings.
        remove_stopwords: Eliminar stopwords.
        remove_accents: Eliminar acentos.

    Returns:
        Lista de listas de tokens.
    """
    result: list[list[str]] = []
    for text in texts:
        cleaned = clean_text(text, remove_accents=remove_accents)
        result.append(tokenize(cleaned, remove_stopwords=remove_stopwords))
    logger.info("Preprocesados %d documentos", len(result))
    return result
