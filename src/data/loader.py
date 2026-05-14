"""Carga de reseñas desde archivos CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("review_id", "product_category", "text", "rating")
DEFAULT_ENCODING: str = "utf-8"


def load_reviews(path: str | Path, encoding: str = DEFAULT_ENCODING) -> pd.DataFrame:
    """Carga un archivo CSV de reseñas y valida sus columnas.

    Args:
        path: Ruta al archivo CSV con las reseñas.
        encoding: Codificación del archivo (por defecto utf-8).

    Returns:
        DataFrame con las reseñas cargadas.

    Raises:
        FileNotFoundError: Si la ruta no existe.
        ValueError: Si faltan columnas requeridas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de reseñas: {path}")

    logger.info("Cargando reseñas desde %s", path)
    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding, exc.object, exc.start, exc.end,
            f"No se pudo decodificar {path} con encoding '{encoding}'. "
            f"Prueba con 'latin-1' o 'utf-8-sig'.",
        ) from exc

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {sorted(missing)}")

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    if df["text"].eq("").any():
        raise ValueError("La columna 'text' contiene valores vacíos")

    df["product_category"] = df["product_category"].fillna("").astype(str).str.strip()
    if df["product_category"].eq("").any():
        raise ValueError("La columna 'product_category' contiene valores vacíos")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    invalid_rating = df["rating"].isna() | ~df["rating"].between(1, 5)
    if invalid_rating.any():
        count = int(invalid_rating.sum())
        raise ValueError(f"La columna 'rating' contiene {count} valores fuera del rango 1-5")
    df["rating"] = df["rating"].astype(int)

    logger.info("Se cargaron %d reseñas", len(df))
    return df


def filter_by_category(
    df: pd.DataFrame,
    category: str | None = None,
) -> pd.DataFrame:
    """Filtra el DataFrame por categoría de producto.

    Args:
        df: DataFrame con las reseñas.
        category: Nombre de la categoría a filtrar; si es None, no filtra.

    Returns:
        DataFrame filtrado.
    """
    if category is None:
        return df
    normalized = df["product_category"].astype(str).str.strip().str.lower()
    mask = normalized == category.strip().lower()
    logger.info("Filtradas %d reseñas para categoría '%s'", mask.sum(), category)
    return df.loc[mask].reset_index(drop=True)
