"""Análisis de sentimiento basado en lexicones para español."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WINDOW: int = 5
NEGATION_WINDOW: int = 3
TOKEN_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
NEGATION_TERMS: frozenset[str] = frozenset({"no", "nunca", "jamás", "tampoco", "ningún", "ninguna"})
INTENSIFIERS: dict[str, float] = {
    "muy": 1.5,
    "súper": 1.7,
    "super": 1.7,
    "extremadamente": 1.8,
    "bastante": 1.3,
    "poco": 0.5,
    "ligeramente": 0.6,
}

DEFAULT_LEXICON: dict[str, float] = {
    "excelente": 1.0,
    "fantástico": 1.0,
    "maravilloso": 1.0,
    "increíble": 0.9,
    "bueno": 0.7,
    "buena": 0.7,
    "buenos": 0.7,
    "buenas": 0.7,
    "bonito": 0.6,
    "bonita": 0.6,
    "bonitos": 0.6,
    "bonitas": 0.6,
    "agradable": 0.6,
    "agradables": 0.6,
    "recomendable": 0.8,
    "perfecto": 1.0,
    "perfecta": 1.0,
    "genial": 0.9,
    "rápido": 0.6,
    "rápida": 0.6,
    "rapido": 0.6,
    "rapida": 0.6,
    "cómodo": 0.6,
    "cómoda": 0.6,
    "cómodos": 0.6,
    "cómodas": 0.6,
    "útil": 0.6,
    "útiles": 0.6,
    "práctico": 0.6,
    "resistente": 0.7,
    "duradero": 0.7,
    "duradera": 0.7,
    "elegante": 0.6,
    "barato": 0.4,
    "económico": 0.5,
    "satisfecho": 0.7,
    "feliz": 0.8,
    "encantado": 0.9,
    "malo": -0.7,
    "mala": -0.7,
    "malos": -0.7,
    "malas": -0.7,
    "pésimo": -1.0,
    "pésima": -1.0,
    "pésimos": -1.0,
    "pésimas": -1.0,
    "horrible": -1.0,
    "horribles": -1.0,
    "terrible": -1.0,
    "feo": -0.6,
    "fea": -0.6,
    "incómodo": -0.6,
    "incómoda": -0.6,
    "lento": -0.6,
    "lenta": -0.6,
    "defectuoso": -0.9,
    "defectuosa": -0.9,
    "roto": -0.9,
    "rota": -0.9,
    "caro": -0.4,
    "cara": -0.4,
    "costoso": -0.5,
    "costosa": -0.5,
    "decepcionante": -0.8,
    "decepcionado": -0.7,
    "frágil": -0.5,
    "inútil": -0.8,
    "deficiente": -0.7,
    "frustrante": -0.7,
    "molesto": -0.6,
    "ruidoso": -0.5,
    "complicado": -0.4,
    "engorroso": -0.5,
    "horrendo": -1.0,
    "mediocre": -0.5,
    "aceptable": 0.2,
    "regular": 0.0,
    "normal": 0.1,
}


def load_lexicon(path: str | Path | None = None) -> dict[str, float]:
    """Carga el lexicón desde un archivo JSON o retorna el por defecto.

    Args:
        path: Ruta a un JSON {palabra: score}; si es None usa DEFAULT_LEXICON.

    Returns:
        Diccionario palabra -> score ∈ [-1, 1].
    """
    if path is None:
        logger.info("Usando lexicón por defecto (%d términos)", len(DEFAULT_LEXICON))
        return dict(DEFAULT_LEXICON)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lexicón no encontrado: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, float] = json.load(fh)
    logger.info("Lexicón cargado desde %s (%d términos)", path, len(data))
    return data


def save_lexicon(lexicon: dict[str, float], path: str | Path) -> None:
    """Persiste un lexicón a JSON.

    Args:
        lexicon: Diccionario palabra -> score.
        path: Ruta de salida.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(lexicon, fh, ensure_ascii=False, indent=2, sort_keys=True)
    logger.info("Lexicón guardado en %s", path)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    chars = [
        char
        for char in normalized
        if not unicodedata.combining(char) or char == "\u0303"
    ]
    return unicodedata.normalize("NFC", "".join(chars))


def _normalize_token(text: str) -> str:
    return _strip_accents(text.strip().lower())


def _normalize_lexicon(lexicon: dict[str, float]) -> dict[str, float]:
    return {_normalize_token(word): float(score) for word, score in lexicon.items()}


def _tokenize_words(text: str) -> list[str]:
    """Tokeniza palabras ignorando puntuación y normalizando minúsculas/tildes."""
    return [_normalize_token(match.group(0)) for match in TOKEN_PATTERN.finditer(text)]


def _find_aspect_spans(tokens: list[str], aspect: str) -> list[tuple[int, int]]:
    """Devuelve spans [inicio, fin) donde aparece el aspecto tokenizado."""
    aspect_tokens = _tokenize_words(aspect)
    if not aspect_tokens:
        return []

    width = len(aspect_tokens)
    spans: list[tuple[int, int]] = []
    for i in range(0, len(tokens) - width + 1):
        if tokens[i:i + width] == aspect_tokens:
            spans.append((i, i + width))
    return spans


def score_aspect_lexicon(
    text: str,
    aspect: str,
    lexicon: dict[str, float] | None = None,
    window: int = DEFAULT_WINDOW,
) -> float:
    """Calcula el sentimiento de un aspecto en un texto usando un lexicón.

    Examina una ventana de `window` palabras a cada lado del aspecto, aplica
    negaciones e intensificadores y promedia.

    Args:
        text: Texto de la reseña.
        aspect: Aspecto cuyo sentimiento se quiere medir.
        lexicon: Lexicón opcional; si es None se usa DEFAULT_LEXICON.
        window: Tamaño de la ventana a cada lado del aspecto.

    Returns:
        Score ∈ [-1, 1]. Devuelve 0.0 si no hay términos sentimentales cerca.
    """
    if lexicon is None:
        lexicon = DEFAULT_LEXICON
    lexicon = _normalize_lexicon(lexicon)
    negation_terms = {_normalize_token(term) for term in NEGATION_TERMS}
    intensifiers = {_normalize_token(term): value for term, value in INTENSIFIERS.items()}

    tokens = _tokenize_words(text)
    spans = _find_aspect_spans(tokens, aspect)
    if not spans:
        return 0.0

    scores: list[float] = []
    for aspect_start, aspect_end in spans:
        start = max(0, aspect_start - window)
        end = min(len(tokens), aspect_end + window)
        local_score = 0.0
        local_count = 0
        for i in range(start, end):
            if aspect_start <= i < aspect_end:
                continue
            tok = tokens[i]
            if tok in lexicon:
                val = lexicon[tok]
                # Negación: revisar los NEGATION_WINDOW tokens previos
                neg_window = tokens[max(0, i - NEGATION_WINDOW):i]
                if any(n in negation_terms for n in neg_window):
                    val = -val
                # Intensificadores: token previo
                if i > 0 and tokens[i - 1] in intensifiers:
                    val *= intensifiers[tokens[i - 1]]
                local_score += val
                local_count += 1
        if local_count > 0:
            scores.append(local_score / local_count)

    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    return max(-1.0, min(1.0, avg))
