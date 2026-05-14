from __future__ import annotations

import pytest


def test_loader_rejects_decimal_ratings(tmp_path):
    pytest.importorskip("pandas")
    from src.data.loader import load_reviews

    csv_path = tmp_path / "reviews.csv"
    csv_path.write_text(
        "review_id,product_category,text,rating\n"
        "r1,electronica,Texto válido,4.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="enteros"):
        load_reviews(csv_path)


def test_annotated_dataset_rejects_unmapped_aspects():
    pd = pytest.importorskip("pandas")
    from src.data.builder import build_annotated_dataset

    df = pd.DataFrame(
        {
            "text": ["El empaque fue correcto."],
            "aspect": ["olor"],
            "label": ["pos"],
        }
    )

    with pytest.raises(ValueError, match="fuera del catálogo"):
        build_annotated_dataset(df)
