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
    df = pd.read_csv(path, encoding=encoding)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {sorted(missing)}")

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
    mask = df["product_category"].str.lower() == category.lower()
    logger.info("Filtradas %d reseñas para categoría '%s'", mask.sum(), category)
    return df.loc[mask].reset_index(drop=True)
