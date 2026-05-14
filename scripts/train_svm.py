"""Entrena el clasificador SVM clásico para ABSA."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.classical.svm_classifier import SVMAspectClassifier
from src.data.builder import build_dataset
from src.data.loader import load_reviews
from src.evaluation.metrics import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena SVM clásico para ABSA")
    parser.add_argument("--data", type=str, default="data/sample/reviews_sample.csv")
    parser.add_argument("--out", type=str, default="models/svm.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_reviews(args.data)
    texts, aspects, labels = build_dataset(df)

    t_train, t_test, a_train, a_test, y_train, y_test = train_test_split(
        texts, aspects, labels, test_size=args.test_size, random_state=args.seed, stratify=labels,
    )

    clf = SVMAspectClassifier()
    clf.fit(t_train, a_train, y_train)

    preds = clf.predict(t_test, a_test).tolist()
    metrics = evaluate_predictions(y_test, preds)
    logger.info("Métricas test: %s", {k: v for k, v in metrics.items() if not isinstance(v, list)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.out)


if __name__ == "__main__":
    main()
