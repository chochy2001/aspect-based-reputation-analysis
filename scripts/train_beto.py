"""Entrena BETO para Aspect-Based Sentiment Analysis."""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-5


def _split_reviews(df, test_size: float, seed: int):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    if "product_id" in df.columns and df["product_id"].nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, groups=df["product_id"]))
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    stratify = df["rating"] if df["rating"].value_counts().min() >= 2 else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)


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

    from src.data.builder import build_best_available_dataset
    from src.data.loader import load_reviews
    from src.evaluation.metrics import evaluate_predictions
    from src.transformers_models.beto import BETOAspectClassifier

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
