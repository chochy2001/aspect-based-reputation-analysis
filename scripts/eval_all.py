"""Evalúa los cuatro enfoques (lexicón, SVM, BETO, LLM) en el mismo conjunto."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


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


def _classification_payload(y_true: list[str], y_pred: list[str], aspects: list[str]) -> dict:
    from src.evaluation.metrics import evaluate_predictions, evaluate_predictions_by_aspect

    metrics = evaluate_predictions(y_true, y_pred)
    metrics["by_aspect"] = evaluate_predictions_by_aspect(y_true, y_pred, aspects)
    return metrics


def evaluate_lexicon(texts: list[str], aspects: list[str], y_true: list[str]) -> dict:
    from src.classical.lexicon import load_lexicon, score_aspect_lexicon
    from src.data.builder import sentiment_label

    lexicon = load_lexicon()
    preds = [sentiment_label(score_aspect_lexicon(t, a, lexicon)) for t, a in zip(texts, aspects)]
    return _classification_payload(y_true, preds, aspects)


def evaluate_svm(model_path: str, texts: list[str], aspects: list[str], y_true: list[str]) -> dict:
    from src.classical.svm_classifier import SVMAspectClassifier

    clf = SVMAspectClassifier.load(model_path)
    preds = clf.predict(texts, aspects).tolist()
    return _classification_payload(y_true, preds, aspects)


def evaluate_beto(model_path: str, texts: list[str], aspects: list[str], y_true: list[str]) -> dict:
    from src.transformers_models.beto import BETOAspectClassifier

    clf = BETOAspectClassifier.load(model_path)
    preds = clf.predict(texts, aspects)
    return _classification_payload(y_true, preds, aspects)


def evaluate_llm(texts: list[str], aspects: list[str], y_true: list[str]) -> dict:
    """Evalúa el enfoque LLM (few-shot prompting) sobre el conjunto de prueba.

    Requiere ANTHROPIC_API_KEY u OPENAI_API_KEY en el entorno.
    """
    try:
        from src.llm.prompting import analyze_with_llm, build_default_client
    except ModuleNotFoundError as exc:
        raise _dependency_error(exc) from exc

    client = build_default_client()
    preds: list[str] = []
    for text, aspect in zip(texts, aspects):
        result = analyze_with_llm(text, [aspect], client)
        preds.append(result.get(aspect, "neu"))
    return _classification_payload(y_true, preds, aspects)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa todos los enfoques")
    parser.add_argument("--data", type=str, default="data/sample/reviews_sample.csv")
    parser.add_argument("--svm", type=str, default="models/svm.pkl")
    parser.add_argument("--beto", type=str, default="models/beto")
    parser.add_argument("--out", type=str, default="reports/eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Evalúa todas las filas del CSV recibido; útil cuando el archivo ya es un split de prueba.",
    )
    parser.add_argument(
        "--eval-llm",
        action="store_true",
        help="Ejecuta evaluación LLM remota. Requiere llave API y puede generar costo.",
    )
    parser.add_argument(
        "--allow-pseudo-smoke",
        action="store_true",
        help="Permite evaluar con pseudo-etiquetas de lexicón solo como prueba funcional.",
    )
    args = parser.parse_args()

    try:
        from src.aspects.extractor import extract_aspects, load_spacy_model
        from src.classical.lexicon import load_lexicon, score_aspect_lexicon
        from src.data.builder import build_best_available_dataset, sentiment_label
        from src.data.loader import load_reviews
        from src.reputation.aggregator import compute_reputation_scores
    except ModuleNotFoundError as exc:
        raise _dependency_error(exc) from exc

    df = load_reviews(args.data)
    if args.no_split:
        test_df = df.reset_index(drop=True)
    else:
        _, test_df = _split_reviews(df, args.test_size, args.seed)
    t_test, a_test, y_test, label_source = build_best_available_dataset(test_df)
    if label_source != "manual":
        if not args.allow_pseudo_smoke:
            raise SystemExit(
                "El CSV no contiene anotaciones manuales. Para una prueba de humo usa "
                "--allow-pseudo-smoke; no reportes esas métricas como resultado final."
            )
        logger.warning(
            "Evaluación con pseudo-etiquetas de lexicón; usar solo como prueba funcional, no como resultado final."
        )

    results: dict[str, dict] = {
        "metadata": {
            "label_source": label_source,
            "mode": "smoke" if label_source != "manual" else "gold",
            "test_size": args.test_size,
            "split": "none" if args.no_split else "internal_train_test",
        }
    }
    results["lexicon"] = evaluate_lexicon(t_test, a_test, y_test)

    if Path(args.svm).exists():
        results["svm"] = evaluate_svm(args.svm, t_test, a_test, y_test)
    else:
        logger.warning("No se encontró modelo SVM en %s", args.svm)

    if Path(args.beto).exists():
        results["beto"] = evaluate_beto(args.beto, t_test, a_test, y_test)
    else:
        logger.warning("No se encontró modelo BETO en %s", args.beto)

    if args.eval_llm:
        results["llm"] = evaluate_llm(t_test, a_test, y_test)
    else:
        logger.info("Saltando evaluación LLM remota; usa --eval-llm para ejecutarla con costo/API.")

    # Reputación agregada de humo (lexicón sobre el conjunto de prueba).
    nlp = load_spacy_model()
    lexicon = load_lexicon()
    predictions = []
    for _, row in test_df.iterrows():
        text = str(row["text"])
        per_review: dict[str, str] = {}
        for asp in extract_aspects(text, nlp):
            per_review[asp] = sentiment_label(score_aspect_lexicon(text, asp, lexicon))
        if per_review:
            predictions.append(per_review)
    results["reputation_scores"] = {
        "source": "lexicon_smoke",
        "scope": "test_split",
        "scores": compute_reputation_scores(predictions),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    logger.info("Resultados guardados en %s", out)


if __name__ == "__main__":
    main()
