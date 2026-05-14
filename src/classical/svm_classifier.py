"""Clasificador SVM con TF-IDF para sentimiento aspect-based."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

DEFAULT_MAX_FEATURES: int = 20000
DEFAULT_NGRAM_RANGE: tuple[int, int] = (1, 2)
DEFAULT_C: float = 1.0


class SVMAspectClassifier:
    """Pipeline TF-IDF + LinearSVC para sentimiento por aspecto.

    El input se forma concatenando el texto y el aspecto:
        "<texto> [ASPECT] <aspecto>"
    """

    ASPECT_TOKEN: str = "[ASPECT]"

    def __init__(
        self,
        max_features: int = DEFAULT_MAX_FEATURES,
        ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
        C: float = DEFAULT_C,
    ) -> None:
        """Inicializa el pipeline.

        Args:
            max_features: Máximo de features TF-IDF.
            ngram_range: Rango de n-gramas.
            C: Parámetro de regularización SVM.
        """
        self.pipeline: Pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                lowercase=True,
            )),
            ("svm", LinearSVC(C=C, class_weight="balanced")),
        ])
        self._fitted: bool = False

    def _format_inputs(self, texts: Iterable[str], aspects: Iterable[str]) -> list[str]:
        text_list = list(texts)
        aspect_list = list(aspects)
        if len(text_list) != len(aspect_list):
            raise ValueError("texts y aspects deben tener la misma longitud")
        return [f"{t} {self.ASPECT_TOKEN} {a}" for t, a in zip(text_list, aspect_list)]

    def fit(
        self,
        texts: list[str],
        aspects: list[str],
        labels: list[str],
    ) -> "SVMAspectClassifier":
        """Entrena el clasificador.

        Args:
            texts: Lista de textos.
            aspects: Lista de aspectos paralelos a `texts`.
            labels: Etiquetas (p.ej. 'pos', 'neg', 'neu').

        Returns:
            self.
        """
        if not (len(texts) == len(aspects) == len(labels)):
            raise ValueError("texts, aspects y labels deben tener la misma longitud")
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Se requieren al menos 2 clases distintas; recibidas: {unique_labels}"
            )
        inputs = self._format_inputs(texts, aspects)
        logger.info("Entrenando SVM con %d ejemplos (%d clases)", len(inputs), len(unique_labels))
        self.pipeline.fit(inputs, labels)
        self._fitted = True
        return self

    def predict(self, texts: list[str], aspects: list[str]) -> np.ndarray:
        """Predice etiquetas de sentimiento.

        Args:
            texts: Lista de textos.
            aspects: Lista de aspectos.

        Returns:
            Array de etiquetas predichas.
        """
        if not self._fitted:
            raise RuntimeError("El modelo no ha sido entrenado")
        inputs = self._format_inputs(texts, aspects)
        return self.pipeline.predict(inputs)

    def save(self, path: str | Path) -> None:
        """Guarda el modelo entrenado a disco usando joblib.

        Args:
            path: Ruta de salida (.joblib o .pkl).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Modelo SVM guardado en %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SVMAspectClassifier":
        """Carga un modelo previamente guardado.

        Args:
            path: Ruta al artefacto serializado.

        Returns:
            Instancia con pipeline cargado.
        """
        path = Path(path)
        pipeline = joblib.load(path)
        instance = cls()
        instance.pipeline = pipeline
        instance._fitted = True
        logger.info("Modelo SVM cargado desde %s", path)
        return instance
