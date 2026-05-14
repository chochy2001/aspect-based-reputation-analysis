"""Entrena BETO para Aspect-Based Sentiment Analysis."""

from __future__ import annotations

import argparse
import logging

from sklearn.model_selection import train_test_split

from src.data.builder import build_dataset
from src.data.loader import load_reviews
from src.evaluation.metrics import evaluate_predictions
from src.transformers_models.beto import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    BETOAspectClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena BETO para ABSA")
    parser.add_argument("--data", type=str, default="data/sample/reviews_sample.csv")
    parser.add_argument("--out", type=str, default="models/beto")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_reviews(args.data)
    texts, aspects, labels = build_dataset(df)

    t_train, t_test, a_train, a_test, y_train, y_test = train_test_split(
        texts, aspects, labels, test_size=args.test_size, random_state=args.seed, stratify=labels,
    )

    clf = BETOAspectClassifier()
    clf.fit(
        t_train,
        a_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    preds = clf.predict(t_test, a_test)
    metrics = evaluate_predictions(y_test, preds)
    logger.info("Métricas test: %s", {k: v for k, v in metrics.items() if not isinstance(v, list)})

    clf.save(args.out)


if __name__ == "__main__":
    main()
