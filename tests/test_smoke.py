"""Pruebas de humo para verificar que los módulos del proyecto cargan y
funcionan en su flujo más básico, sin dependencias pesadas (no torch, no spaCy).
"""

from __future__ import annotations

import pytest

from src.aspects.extractor import (
    extract_category_mentions,
    map_to_category,
)
from src.classical.lexicon import (
    DEFAULT_LEXICON,
    load_lexicon,
    score_aspect_lexicon,
)
from src.data.builder import sentiment_label
from src.data.preprocessing import clean_text, tokenize
from src.evaluation.metrics import evaluate_predictions, evaluate_reputation
from src.reputation.aggregator import (
    aspect_score_summary,
    compute_global_reputation,
    compute_reputation_scores,
)


def test_load_default_lexicon_has_known_terms():
    lexicon = load_lexicon()
    assert lexicon["excelente"] == 1.0
    assert lexicon["pésimo"] == -1.0
    assert len(lexicon) >= 40


def test_score_aspect_positive_context():
    text = "La batería es excelente y la calidad muy buena."
    assert score_aspect_lexicon(text, "batería") > 0
    assert score_aspect_lexicon(text, "calidad") > 0


def test_score_aspect_negation_inverts_polarity():
    text_pos = "La pantalla es buena"
    text_neg = "La pantalla no es buena"
    assert score_aspect_lexicon(text_pos, "pantalla") > 0
    assert score_aspect_lexicon(text_neg, "pantalla") < 0


def test_score_aspect_missing_aspect_returns_zero():
    assert score_aspect_lexicon("excelente producto", "pantalla") == 0.0


def test_sentiment_label_thresholds():
    assert sentiment_label(0.5) == "pos"
    assert sentiment_label(-0.5) == "neg"
    assert sentiment_label(0.05) == "neu"


def test_clean_text_removes_urls_and_punctuation():
    cleaned = clean_text("Mira esto https://x.com !!!", lowercase=True)
    assert "https" not in cleaned
    assert "!" not in cleaned


def test_tokenize_returns_words():
    assert tokenize("hola mundo") == ["hola", "mundo"]


def test_extract_category_mentions_finds_known_keywords():
    text = "el envío fue rápido y la calidad excelente"
    cats = {c for c, _ in extract_category_mentions(text)}
    assert "envío" in cats
    assert "calidad" in cats


def test_map_to_category():
    assert map_to_category("batería") == "durabilidad"
    assert map_to_category("precio") == "precio"
    assert map_to_category("desconocido") is None


def test_compute_reputation_aggregates_positive():
    preds = [{"calidad": "pos"}, {"calidad": "pos"}, {"calidad": "pos"}]
    scores = compute_reputation_scores(preds)
    assert "calidad" in scores
    assert scores["calidad"] > 2.5  # debe ser positivo


def test_compute_reputation_empty_returns_empty():
    assert compute_reputation_scores([]) == {}


def test_aspect_score_summary_labels():
    summary = aspect_score_summary({"calidad": 4.5, "precio": 1.0})
    assert summary["calidad"] == "muy positivo"
    assert summary["precio"] == "negativo"


def test_global_reputation_average():
    score = compute_global_reputation({"calidad": 4.0, "precio": 2.0})
    assert score == pytest.approx(3.0)


def test_evaluate_predictions_basic_metrics():
    metrics = evaluate_predictions(["pos", "neg", "neu"], ["pos", "neg", "neu"])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_evaluate_predictions_mismatch_lengths_raises():
    with pytest.raises(ValueError):
        evaluate_predictions(["pos"], ["pos", "neg"])


def test_evaluate_reputation_no_common_aspects():
    metrics = evaluate_reputation({"a": 1.0}, {"b": 2.0})
    assert metrics["n_aspects"] == 0
