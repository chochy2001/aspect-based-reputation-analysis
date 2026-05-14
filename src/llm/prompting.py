"""Few-shot prompting con Anthropic Claude (preferido) u OpenAI como fallback."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

VALID_LABELS: tuple[str, ...] = ("pos", "neg", "neu")
DEFAULT_ANTHROPIC_MODEL: str = "claude-sonnet-4-5-20250929"
DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
DEFAULT_MAX_TOKENS: int = 512
DEFAULT_TEMPERATURE: float = 0.0

Provider = Literal["anthropic", "openai"]

SYSTEM_PROMPT: str = (
    "Eres un experto en análisis de sentimiento sobre reseñas de productos en español. "
    "Tu tarea es determinar el sentimiento (pos, neg, neu) que la reseña expresa "
    "hacia cada aspecto solicitado. Responde ÚNICAMENTE con un JSON válido."
)

FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "text": "La batería del móvil dura todo el día pero la pantalla se ve algo opaca.",
        "aspects": ["batería", "pantalla"],
        "answer": {"batería": "pos", "pantalla": "neg"},
    },
    {
        "text": "El envío llegó rápido y el producto está perfecto, muy contento con la compra.",
        "aspects": ["envío", "producto"],
        "answer": {"envío": "pos", "producto": "pos"},
    },
    {
        "text": "El precio es aceptable, la calidad es regular y el servicio al cliente fue muy malo.",
        "aspects": ["precio", "calidad", "servicio"],
        "answer": {"precio": "neu", "calidad": "neu", "servicio": "neg"},
    },
    {
        "text": "El libro llegó bien empacado y la lectura es entretenida.",
        "aspects": ["envío", "calidad", "precio"],
        "answer": {"envío": "pos", "calidad": "pos", "precio": "neu"},
    },
]


def _build_user_prompt(text: str, aspects: list[str]) -> str:
    """Construye el prompt de usuario con ejemplos few-shot."""
    parts: list[str] = ["Ejemplos:\n"]
    for ex in FEW_SHOT_EXAMPLES:
        parts.append(f"Reseña: \"{ex['text']}\"")
        parts.append(f"Aspectos: {ex['aspects']}")
        parts.append(f"Respuesta: {json.dumps(ex['answer'], ensure_ascii=False)}\n")
    parts.append("Ahora analiza la siguiente reseña:")
    parts.append(f"Reseña: \"{text}\"")
    parts.append(f"Aspectos: {aspects}")
    parts.append("Respuesta:")
    return "\n".join(parts)


def _extract_first_json_object(content: str) -> str | None:
    """Extrae el primer objeto JSON balanceado de una cadena."""
    start = content.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(content)):
        ch = content[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return content[start:idx + 1]
    return None


def _parse_response(content: str, aspects: list[str]) -> dict[str, str]:
    """Extrae un JSON {aspecto: sentimiento} de la respuesta del modelo.

    Args:
        content: Texto devuelto por el LLM.
        aspects: Aspectos esperados.

    Returns:
        Diccionario {aspecto: 'pos'|'neg'|'neu'}. Valores inválidos -> 'neu'.
    """
    json_text = _extract_first_json_object(content)
    if json_text is None:
        logger.warning("No se encontró JSON en la respuesta del LLM")
        return {a: "neu" for a in aspects}
    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError:
        logger.warning("JSON inválido en la respuesta del LLM")
        return {a: "neu" for a in aspects}

    result: dict[str, str] = {}
    for aspect in aspects:
        value = raw.get(aspect, "neu")
        if isinstance(value, str) and value.lower() in VALID_LABELS:
            result[aspect] = value.lower()
        else:
            result[aspect] = "neu"
    return result


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(Exception),
)
def _call_anthropic(client, text: str, aspects: list[str], model: str) -> str:
    """Llama a la API de Anthropic con reintentos y backoff exponencial."""
    user_prompt = _build_user_prompt(text, aspects)
    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(Exception),
)
def _call_openai(client, text: str, aspects: list[str], model: str) -> str:
    """Llama a la API de OpenAI con reintentos y backoff exponencial."""
    user_prompt = _build_user_prompt(text, aspects)
    response = client.chat.completions.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def _detect_provider(client) -> Provider:
    """Detecta el proveedor a partir del tipo del cliente."""
    cls = type(client).__name__.lower()
    if "anthropic" in cls:
        return "anthropic"
    if "openai" in cls:
        return "openai"
    if hasattr(client, "messages"):
        return "anthropic"
    if hasattr(client, "chat"):
        return "openai"
    raise ValueError("No se pudo detectar el proveedor del cliente LLM")


def analyze_with_llm(
    text: str,
    aspects: list[str],
    client,
    model: str | None = None,
) -> dict[str, str]:
    """Analiza el sentimiento por aspecto usando un LLM con few-shot prompting.

    Args:
        text: Texto de la reseña en español.
        aspects: Aspectos a evaluar.
        client: Cliente Anthropic (preferido) u OpenAI ya autenticado.
        model: Modelo a usar; si es None se elige uno por defecto según el proveedor.

    Returns:
        Diccionario {aspecto: 'pos'|'neg'|'neu'}.
    """
    if not aspects:
        return {}

    provider = _detect_provider(client)
    logger.info("Llamando LLM (%s) para %d aspectos", provider, len(aspects))

    if provider == "anthropic":
        chosen = model or DEFAULT_ANTHROPIC_MODEL
        content = _call_anthropic(client, text, aspects, chosen)
    else:
        chosen = model or DEFAULT_OPENAI_MODEL
        content = _call_openai(client, text, aspects, chosen)

    return _parse_response(content, aspects)


def build_default_client():
    """Construye el cliente LLM por defecto.

    Prioridad: Anthropic si ANTHROPIC_API_KEY está definida; OpenAI en caso contrario.

    Returns:
        Cliente listo para `analyze_with_llm`.

    Raises:
        RuntimeError: Si no hay credenciales disponibles.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic

        logger.info("Usando proveedor Anthropic")
        return Anthropic()
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        logger.info("Usando proveedor OpenAI")
        return OpenAI()
    raise RuntimeError(
        "No se encontró ANTHROPIC_API_KEY ni OPENAI_API_KEY en el entorno."
    )
