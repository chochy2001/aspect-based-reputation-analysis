# aspect-based-reputation-analysis

**Análisis de Reputación Basado en Aspectos sobre Reseñas de Productos en Español**

Trabajo final de la materia *Análisis y Procesamiento Inteligente de Textos* (APIT), UNAM, Facultad de Ingeniería.

---

## Autores

- Gómez Vázquez Juan Pablo
- Martínez Miranda Juan Carlos
- Salgado Miranda Jorge

**Profesor:** M. en C. Octavio Augusto Sánchez Velázquez
**Semestre:** 2026-2

---

## Descripción

Este repositorio implementa y compara tres enfoques para Análisis Basado en Aspectos (ABSA) aplicado a la construcción de puntuaciones de reputación a partir de reseñas de productos en español:

1. **Enfoque clásico**, lexicón en español + clasificador SVM con TF-IDF.
2. **Transformers**, BETO (`dccuchile/bert-base-spanish-wwm-uncased`) ajustado con la técnica de *auxiliary sentence* de Sun et al. (2019).
3. **LLMs con prompting**, *few-shot prompting* sobre Anthropic Claude u OpenAI.

Cada modelo produce predicciones de sentimiento por aspecto, que se agregan por dimensión canónica: calidad, precio, envío, durabilidad y atención.

## Estado de reproducibilidad

El repositorio contiene tres niveles de artefactos:

1. **Código fuente, notebooks y pruebas automatizadas** del flujo completo (extracción de aspectos, clasificación con tres enfoques, agregación reputacional, métricas).
2. **CSV ficticio** en `data/sample/reviews_sample.csv` para pruebas de humo y validación del flujo en CI.
3. **Resultados experimentales del piloto** reportados en `docs/paper.md` (Tablas 1-5), generados ejecutando este mismo código sobre un corpus de 1 500 reseñas en español anotadas manualmente por los autores. Por consideraciones de licencia y privacidad, el corpus anotado no se versiona en este repositorio público; se distribuye bajo solicitud al equipo.

Para reproducir los números reportados en el paper se requiere disponer del corpus anotado con las columnas indicadas en la sección "Reproducir experimentos completos". Para auditar el flujo sin acceso al corpus, usar el sample ficticio con `--allow-pseudo-smoke`. La lista de pasos de auditoría se documenta en [`docs/reproducibility_checklist.md`](docs/reproducibility_checklist.md).

## Estructura del repositorio

```
.
├── docs/ # Paper académico (Markdown + BibTeX)
│ ├── paper.md
│ ├── metadata.yaml
│ └── references.bib
├── src/ # Código fuente
│ ├── data/ # Carga y preprocesamiento
│ ├── aspects/ # Extracción de aspectos
│ ├── classical/ # Lexicón + SVM
│ ├── transformers_models/ # BETO fine-tuning
│ ├── llm/ # Prompting con LLMs
│ ├── reputation/ # Agregación a puntuaciones 0-5
│ └── evaluation/ # Métricas
├── notebooks/ # Jupyter notebooks por experimento
├── scripts/ # Scripts de entrenamiento y evaluación
└── data/ # Datasets (instrucciones de descarga en data/README.md)
```

## Instalación

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . # recomendado para usar el paquete desde notebooks
python -m spacy download es_core_news_lg
```

> Los scripts insertan la raíz del repositorio en `sys.path`, por lo que pueden ejecutarse desde una copia local. El modo editable sigue siendo recomendable para notebooks, autocompletado y validación en entornos externos.

## Validación rápida

El repositorio incluye datos ficticios en `data/sample/reviews_sample.csv` para pruebas de humo. Si el CSV no incluye columnas de anotación (`aspect`/`aspecto` y `label`/`sentiment`/`polaridad`), los scripts generan pseudo-etiquetas con el lexicón y lo reportan como advertencia. Esas métricas sirven para validar el flujo, no como resultado experimental final.

```bash
python -m pytest -q
python scripts/train_svm.py --data data/sample/reviews_sample.csv --out /tmp/apit-svm.pkl --allow-pseudo-smoke
python scripts/eval_all.py --data data/sample/reviews_sample.csv --svm /tmp/apit-svm.pkl --beto /tmp/no-beto --out /tmp/apit-eval.json --allow-pseudo-smoke
```

## Reproducir experimentos completos

Para resultados reportables se necesita un CSV anotado manualmente o por consenso con estas columnas mínimas:

| Columna | Uso |
|---|---|
| `review_id` | Identificador de reseña |
| `product_category` | Categoría del producto |
| `text` | Texto de la reseña |
| `rating` | Calificación 1-5 |
| `aspect` o `aspecto` | Aspecto anotado |
| `label`, `sentiment` o `polaridad` | `pos`, `neg` o `neu` |

Con datos reales:

```bash
# Experimento 1, Clásico
python scripts/train_svm.py --data data/processed/reviews_train.csv --out models/svm.pkl --train-all

# Experimento 2, BETO
python scripts/train_beto.py --data data/processed/reviews_train.csv --out models/beto/ --train-all

# Evaluación comparativa
export ANTHROPIC_API_KEY=...
python scripts/eval_all.py --data data/processed/reviews_test.csv --out reports/eval_results.json --no-split --eval-llm
```

Si no se desea llamar a un proveedor externo, omitir `--eval-llm`; el evaluador saltará esa parte sin consumir API.

## Documento académico

El reporte completo se encuentra en [`docs/paper.md`](docs/paper.md). Se puede compilar a PDF con pandoc y el motor Typst (más ligero que LaTeX y sin dependencias del sistema):

```bash
# Requisitos: pandoc 3.0+ y typst 0.13+
brew install pandoc typst   # macOS
# o equivalentes en Linux/Windows

cd docs/
pandoc paper.md \
  --pdf-engine=typst \
  --bibliography=references.bib \
  --citeproc \
  --toc --number-sections \
  -o paper.pdf
```

Resultado esperado: `docs/paper.pdf` con tabla de contenido y bibliografía numerada.

## Licencia

Código y documentación bajo licencia MIT. Material académico con fines educativos. UNAM 2026.
