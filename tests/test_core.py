import unittest

from src.aspects.extractor import extract_aspects, map_to_category
from src.classical.lexicon import score_aspect_lexicon
from src.data.preprocessing import clean_text, strip_accents
from src.reputation.aggregator import (
    compute_global_reputation,
    compute_product_reputation_scores,
    compute_reputation_scores,
)


class PreprocessingTests(unittest.TestCase):
    def test_strip_accents_preserves_enye(self) -> None:
        self.assertEqual(strip_accents("Ñandú y piñata"), "Ñandu y piñata")

    def test_clean_text_keeps_uppercase_when_requested(self) -> None:
        self.assertEqual(clean_text("Ñandú ABC", lowercase=False, remove_accents=True), "Ñandu ABC")


class AspectExtractionTests(unittest.TestCase):
    def test_keyword_extraction_works_without_spacy_model(self) -> None:
        aspects = extract_aspects("El envío fue rápido y la batería dura mucho.", nlp=None)
        self.assertIn("envío", aspects)
        self.assertIn("batería", aspects)

    def test_map_to_canonical_category_normalizes_accents(self) -> None:
        self.assertEqual(map_to_category("envio"), "envío")
        self.assertEqual(map_to_category("atencion"), "atención")


class LexiconTests(unittest.TestCase):
    def test_aspect_matching_ignores_punctuation(self) -> None:
        score = score_aspect_lexicon("La batería, es excelente.", "batería")
        self.assertGreater(score, 0)

    def test_multiword_aspect_matching(self) -> None:
        score = score_aspect_lexicon("El servicio al cliente fue muy malo.", "servicio al cliente")
        self.assertLess(score, 0)


class ReputationTests(unittest.TestCase):
    def test_reputation_scores_are_clamped(self) -> None:
        scores = compute_reputation_scores([{"calidad": "pos"}], prior=5.0)
        self.assertLessEqual(scores["calidad"], 5.0)
        self.assertGreaterEqual(scores["calidad"], 0.0)

    def test_invalid_sentiment_label_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            compute_reputation_scores([{"calidad": "positivo"}])

    def test_confidence_weighted_product_scores(self) -> None:
        records = [
            {"product_id": "p1", "aspect": "precio", "sentiment": "pos", "confidence": 0.9},
            {"product_id": "p1", "aspect": "precio", "sentiment": "neg", "confidence": 0.1},
            {"product_id": "p2", "aspect": "envío", "sentiment": "neg", "confidence": 1.0},
        ]
        scores = compute_product_reputation_scores(records)
        self.assertGreater(scores["p1"]["precio"], 2.5)
        self.assertLess(scores["p2"]["envío"], 2.5)

    def test_global_reputation_is_clamped(self) -> None:
        self.assertEqual(compute_global_reputation({"calidad": 9.0}), 5.0)


if __name__ == "__main__":
    unittest.main()
