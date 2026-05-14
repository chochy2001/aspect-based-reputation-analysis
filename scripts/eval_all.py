"""Evalúa los cuatro enfoques (lexicón, SVM, BETO, LLM) en el mismo conjunto."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def _split_reviews(df, test_size: float, seed: int):
    from sklearn.model_selection import train_test_split

    stratify = df["rating"] if df["rating"].value_counts().min() >= 2 else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)


def evaluate_lexicon(texts: list[str], aspects: list[str], y_true: list[str]) -> dict:
    from src.classical.lexicon import load_lexicon, score_aspect_lexicon
    from src.data.builder import sentiment_label
    from src.evaluation.metrics import evaluate_predictions

    lexicon = load_lexicon()
    preds = [sentiment_label(score_aspect_lexicon(t, a, lexicon)) for t, a in zip(texts, aspects)]
    return evaluate_predictions(y_true, preds)


def evaluate_svm(model_path: str, texts: list[str], aspects: list[str], y_true: list[str]) -> dict:
    from src.classical.svm_classifier import SVMAspectClassifier
    from src.evaluation.metrics import evaluate_predictions

    clf = SVMAspectClassifier.load(model_path)
    preds = clf.predict(texts, aspects).tolist()
    return evaluate_predictions(y_true, preds)


def evaluate_beto(model_path: str, texts: list[str], aspects: list[str], y_true: list[str]) -> dict:
    from src.evaluation.metrics import evaluate_predictions
    from src.transformers_models.beto import BETOAspectClassifier

    clf = BETOAspectClassifier.load(model_path)
    preds = clf.predict(texts, aspects)
    return evaluate_predictions(y_true, preds)


def evaluate_llm(texts: list[str], aspects: list[str], y_true: list[str]) -> dict:
    """Evalúa el enfoque LLM (few-shot prompting) sobre el conjunto de prueba.

    Requiere ANTHROPIC_API_KEY u OPENAI_API_KEY en el entorno.
    """
    from src.llm.prompting import analyze_with_llm, build_default_client

    client = build_default_client()
    preds: list[str] = []
    for text, aspect in zip(texts, aspects):
        result = analyze_with_llm(text, [aspect], client)
        preds.append(result.get(aspect, "neu"))
    return evaluate_predictions(y_true, preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa todos los enfoques")
    parser.add_argument("--data", type=str, default="data/sample/reviews_sample.csv")
    parser.add_argument("--svm", type=str, default="models/svm.pkl")
    parser.add_argument("--beto", type=str, default="models/beto")
    parser.add_argument("--out", type=str, default="reports/eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    from src.aspects.extractor import extract_aspects, load_spacy_model
    from src.classical.lexicon import load_lexicon, score_aspect_lexicon
    from src.data.builder import build_best_available_dataset, sentiment_label
    from src.data.loader import load_reviews
    from src.reputation.aggregator import compute_reputation_scores

    df = load_reviews(args.data)
    _, test_df = _split_reviews(df, args.test_size, args.seed)
    t_test, a_test, y_test, label_source = build_best_available_dataset(test_df)
    if label_source != "manual":
        logger.warning(
            "Evaluación con pseudo-etiquetas de lexicón; usar solo como prueba funcional, no como resultado final."
        )

    results: dict[str, dict] = {"metadata": {"label_source": label_source, "test_size": args.test_size}}
    results["lexicon"] = evaluate_lexicon(t_test, a_test, y_test)

    if Path(args.svm).exists():
        results["svm"] = evaluate_svm(args.svm, t_test, a_test, y_test)
    else:
        logger.warning("No se encontró modelo SVM en %s", args.svm)

    if Path(args.beto).exists():
        results["beto"] = evaluate_beto(args.beto, t_test, a_test, y_test)
    else:
        logger.warning("No se encontró modelo BETO en %s", args.beto)

    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
        try:
            results["llm"] = evaluate_llm(t_test, a_test, y_test)
        except Exception as exc:
            logger.warning("Evaluación LLM falló: %s", exc)
    else:
        logger.info("Saltando evaluación LLM (no hay ANTHROPIC_API_KEY ni OPENAI_API_KEY)")

    # Reputación agregada (usando lexicón como ejemplo)
    nlp = load_spacy_model()
    lexicon = load_lexicon()
    predictions = []
    for _, row in df.iterrows():
        text = str(row["text"])
        per_review: dict[str, str] = {}
        for asp in extract_aspects(text, nlp):
            per_review[asp] = sentiment_label(score_aspect_lexicon(text, asp, lexicon))
        if per_review:
            predictions.append(per_review)
    results["reputation_scores"] = compute_reputation_scores(predictions)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    logger.info("Resultados guardados en %s", out)


if __name__ == "__main__":
    main()
