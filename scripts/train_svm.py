"""Entrena el clasificador SVM clásico para ABSA."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def _split_reviews(df, test_size: float, seed: int):
    from sklearn.model_selection import train_test_split

    stratify = df["rating"] if df["rating"].value_counts().min() >= 2 else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena SVM clásico para ABSA")
    parser.add_argument("--data", type=str, default="data/sample/reviews_sample.csv")
    parser.add_argument("--out", type=str, default="models/svm.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.classical.svm_classifier import SVMAspectClassifier
    from src.data.builder import build_best_available_dataset
    from src.data.loader import load_reviews
    from src.evaluation.metrics import evaluate_predictions

    df = load_reviews(args.data)
    train_df, test_df = _split_reviews(df, args.test_size, args.seed)

    t_train, a_train, y_train, train_source = build_best_available_dataset(train_df)
    t_test, a_test, y_test, test_source = build_best_available_dataset(test_df)
    if train_source != "manual" or test_source != "manual":
        logger.warning(
            "Entrenamiento/evaluación con pseudo-etiquetas de lexicón; no reportar estas métricas como verdad terreno."
        )
    if len(set(y_train)) < 2:
        raise ValueError("El conjunto de entrenamiento necesita al menos dos clases de sentimiento")

    clf = SVMAspectClassifier()
    clf.fit(t_train, a_train, y_train)

    preds = clf.predict(t_test, a_test).tolist()
    metrics = evaluate_predictions(y_test, preds)
    logger.info("Métricas test: %s", {k: v for k, v in metrics.items() if not isinstance(v, list)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.out)


if __name__ == "__main__":
    main()
