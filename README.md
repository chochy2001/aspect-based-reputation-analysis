# aspect-based-reputation-analysis

**Análisis de Reputación Basado en Aspectos sobre Reseñas de Productos en Español**

Trabajo final de la materia *Análisis y Procesamiento Inteligente de Textos* (APIT) — UNAM, Facultad de Ingeniería.

---

## Autores

- Gómez Vázquez Juan Pablo
- Martínez Miranda Juan Carlos
- Salgado Miranda Jorge

**Profesor:** M. en C. Octavio Augusto Sánchez Velázquez
**Semestre:** 2026-2

---

## Descripción

Este repositorio implementa y compara tres enfoques para Análisis Basado en Aspectos (ABSA) aplicado a la construcción de scores de reputación a partir de reseñas de productos en español (Amazon / MercadoLibre):

1. **Enfoque clásico** — Lexicones (Senti-py / VADER traducido) + clasificador SVM con TF-IDF.
2. **Transformers** — BETO (`dccuchile/bert-base-spanish-wwm-uncased`) fine-tuned con la técnica de *auxiliary sentence* de Sun et al. (2019).
3. **LLMs con prompting** — Few-shot prompting sobre Anthropic Claude / OpenAI GPT.

Cada modelo produce predicciones de sentimiento por aspecto, las cuales se agregan en *scores de reputación* por dimensión (calidad, precio, envío, durabilidad, atención).

## Estructura del repositorio

```
.
├── docs/                # Paper académico (Markdown + LaTeX + BibTeX)
│   ├── paper.md
│   ├── paper.tex
│   └── references.bib
├── src/                 # Código fuente
│   ├── data/            # Carga y preprocesamiento
│   ├── aspects/         # Extracción de aspectos
│   ├── classical/       # Lexicón + SVM
│   ├── transformers_models/   # BETO fine-tuning
│   ├── llm/             # Prompting con LLMs
│   ├── reputation/      # Agregación a scores 0-5
│   └── evaluation/      # Métricas
├── notebooks/           # Jupyter notebooks por experimento
├── scripts/             # Scripts de entrenamiento y evaluación
└── data/                # Datasets (instrucciones de descarga en data/README.md)
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                    # registra el paquete src/ como editable
python -m spacy download es_core_news_lg
```

> El paso `pip install -e .` es necesario para que los scripts en `scripts/` puedan
> importar `from src.data.builder import ...` sin manipular `PYTHONPATH`.

## Reproducir experimentos

```bash
# Experimento 1 — Clásico
python scripts/train_svm.py --data data/reviews_train.csv --out models/svm.pkl

# Experimento 2 — BETO
python scripts/train_beto.py --data data/reviews_train.csv --out models/beto/

# Experimento 3 — LLM
export ANTHROPIC_API_KEY=...
python scripts/eval_all.py --data data/reviews_test.csv
```

## Documento académico

El reporte completo (siguiendo la estructura solicitada en clase: introducción, motivación, marco teórico, aparato crítico, setup experimental, experimentos, resultados, análisis, discusión) se encuentra en [`docs/paper.md`](docs/paper.md) y compilable a PDF vía `docs/paper.tex`.

```bash
cd docs/
pandoc paper.md -o paper.pdf --bibliography=references.bib --citeproc
```

## Licencia

Material académico con fines educativos. UNAM 2026.
