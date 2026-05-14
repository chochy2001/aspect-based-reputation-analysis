"""Entrena BETO para Aspect-Based Sentiment Analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-5


def _dependency_error(exc: ModuleNotFoundError) -> SystemExit:
    return SystemExit(
        f"Falta la dependencia '{exc.name}'. Instala el entorno con: "
        "python -m pip install -r requirements.txt && python -m pip install -e ."
    )


def _split_reviews(df, test_size: float, seed: int):
    try:
        from sklearn.model_selection import GroupShuffleSplit, train_test_split
    except ModuleNotFoundError as exc:
        raise _dependency_error(exc) from exc

    group_column = None
    if "product_id" in df.columns and df["product_id"].nunique() > 1:
        group_column = "product_id"
    elif "review_id" in df.columns and df["review_id"].nunique() > 1:
        group_column = "review_id"

    if group_column is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, groups=df[group_column]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        overlap = set(train_df[group_column].astype(str)) & set(test_df[group_column].astype(str))
        if overlap:
            raise RuntimeError(f"Fuga de grupos entre train/test en {group_column}: {sorted(overlap)[:5]}")
        return train_df, test_df
    stratify = df["rating"] if df["rating"].value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena BETO para ABSA")
    parser.add_argument("--data", type=str, default="data/sample/reviews_sample.csv")
    parser.add_argument("--out", type=str, default="models/beto")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Entrena con todas las filas del CSV y omite evaluación interna; útil cuando ya existe un split externo.",
    )
    parser.add_argument(
        "--allow-pseudo-smoke",
        action="store_true",
        help="Permite entrenar/evaluar con pseudo-etiquetas de lexicón solo como prueba funcional.",
    )
    args = parser.parse_args()

    try:
        from src.data.builder import build_best_available_dataset
        from src.data.loader import load_reviews
        from src.evaluation.metrics import evaluate_predictions
        from src.transformers_models.beto import BETOAspectClassifier
    except ModuleNotFoundError as exc:
        raise _dependency_error(exc) from exc

    df = load_reviews(args.data)
    if args.train_all:
        train_df = df.reset_index(drop=True)
        test_df = None
    else:
        train_df, test_df = _split_reviews(df, args.test_size, args.seed)

    t_train, a_train, y_train, train_source = build_best_available_dataset(train_df)
    sources = [train_source]
    if test_df is not None:
        t_test, a_test, y_test, test_source = build_best_available_dataset(test_df)
        sources.append(test_source)

    if any(source != "manual" for source in sources):
        if not args.allow_pseudo_smoke:
            raise SystemExit(
                "El CSV no contiene anotaciones manuales. Para una prueba de humo usa "
                "--allow-pseudo-smoke; no reportes esas métricas como resultado final."
            )
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
        seed=args.seed,
    )

    if test_df is not None:
        preds = clf.predict(t_test, a_test)
        metrics = evaluate_predictions(y_test, preds)
        logger.info("Métricas test: %s", {k: v for k, v in metrics.items() if not isinstance(v, list)})
    else:
        logger.info("Entrenamiento completado sobre todas las filas; evaluación interna omitida por --train-all.")

    clf.save(args.out)


if __name__ == "__main__":
    main()
