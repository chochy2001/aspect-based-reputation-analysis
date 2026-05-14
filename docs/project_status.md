# Estado del proyecto y continuidad

Fecha de actualización: 14 de mayo de 2026.

Este documento resume qué contiene el proyecto, qué está verificado, qué falta y cómo continuar sin perder trazabilidad.

## Estado general

El proyecto está integrado en `main` y contiene el flujo completo de análisis de reputación basado en aspectos para reseñas en español:

- extracción y normalización de aspectos;
- clasificación de polaridad por lexicón, SVM, BETO y LLM;
- agregación de sentimientos a puntuaciones 0 a 5;
- evaluación con métricas de clasificación y reputación;
- paper académico, notebooks, scripts y pruebas automatizadas.

El repositorio público incluye un dataset ficticio de prueba en `data/sample/reviews_sample.csv`. El README y el paper documentan que el corpus piloto anotado no se versiona por restricciones de licencia y privacidad. El sample público sirve para pruebas de humo, no para recalcular por sí solo las métricas del piloto.

## Artefactos disponibles

### Documentación

- `README.md`: instalación, estructura, validación rápida y reproducción de experimentos.
- `docs/paper.md`: fuente del reporte académico.
- `docs/paper.pdf`: PDF generado localmente desde `docs/paper.md` (ignorado por git).
- `docs/references.bib`: bibliografía citada.
- `docs/metadata.yaml`: metadatos alternos para Pandoc.
- `docs/reproducibility_checklist.md`: checklist para corridas reportables.
- `docs/project_status.md`: este documento de continuidad.
- `data/README.md`: formato esperado de datos, fuentes sugeridas y trazabilidad.

### Código

- `src/data/`: carga, validación y construcción de datasets.
- `src/aspects/`: extracción de aspectos con spaCy y fallback por palabras clave.
- `src/classical/`: lexicón y SVM con TF-IDF.
- `src/transformers_models/`: BETO con formato reseña-aspecto.
- `src/llm/`: prompting con Anthropic u OpenAI.
- `src/reputation/`: agregación reputacional con suavizado.
- `src/evaluation/`: métricas de clasificación, reputación y reporte por aspecto.

### Scripts

- `scripts/train_svm.py`: entrenamiento del baseline clásico.
- `scripts/train_beto.py`: entrenamiento de BETO.
- `scripts/eval_all.py`: evaluación comparativa.

Flags importantes:

- `--allow-pseudo-smoke`: permite pseudo-etiquetas solo para pruebas funcionales.
- `--train-all`: entrena con todo el CSV cuando el split ya está preparado.
- `--no-split`: evalúa todo el CSV recibido cuando ya es test.
- `--eval-llm`: activa llamadas remotas a LLM.

### Notebooks

- `01_exploracion_datos.ipynb`
- `02_baseline_clasico.ipynb`
- `03_beto_finetuning.ipynb`
- `04_llm_prompting.ipynb`
- `05_comparativa_resultados.ipynb`

### Pruebas

- `tests/test_core.py`
- `tests/test_smoke.py`
- `tests/test_cli_and_docs.py`
- `tests/test_data_validation.py`

## Validación más reciente

Comandos ejecutados localmente:

```bash
python3 -m pytest -q
python3 -m compileall src scripts tests
git diff --check
cd docs && pandoc paper.md --pdf-engine=typst --bibliography=references.bib --citeproc --toc --number-sections -o paper.pdf
cd docs && pandoc paper.md --metadata-file=metadata.yaml --pdf-engine=typst --citeproc -o /tmp/apit-paper-metadata.pdf
```

Resultado:

```text
31 passed, 2 skipped
compileall OK
git diff --check OK
Pandoc + Typst OK
29 referencias BibTeX citadas, 0 faltantes, 0 sobrantes
```

Los dos tests omitidos dependen de paquetes de datos no instalados en el Python global. En el entorno completo creado con `requirements.txt`, esos casos deben ejecutarse.

## Comandos de continuidad

### Instalar entorno

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m spacy download es_core_news_lg
```

### Validar sin corpus real

```bash
python -m pytest -q
python scripts/train_svm.py --data data/sample/reviews_sample.csv --out /tmp/apit-svm.pkl --allow-pseudo-smoke
python scripts/eval_all.py --data data/sample/reviews_sample.csv --svm /tmp/apit-svm.pkl --beto /tmp/no-beto --out /tmp/apit-eval.json --allow-pseudo-smoke
```

### Ejecutar con splits externos

```bash
python scripts/train_svm.py --data data/processed/reviews_train.csv --out models/svm.pkl --train-all
python scripts/train_beto.py --data data/processed/reviews_train.csv --out models/beto/ --train-all
python scripts/eval_all.py --data data/processed/reviews_test.csv --out reports/eval_results.json --no-split
```

### Ejecutar LLM

```bash
export ANTHROPIC_API_KEY=...
python scripts/eval_all.py --data data/processed/reviews_test.csv --out reports/eval_results.json --no-split --eval-llm
```

## Reglas para no romper el proyecto

- No reportar métricas de `--allow-pseudo-smoke` como resultados científicos.
- No activar `--eval-llm` sin revisar privacidad, costo y credenciales.
- No volver a partir un archivo que ya es `test`; usar `--no-split`.
- No entrenar con split preparado sin `--train-all`.
- Mantener `review_id` y, si existe, `product_id` para evitar fuga.
- Registrar comando exacto, semilla, fecha, entorno y fuente de datos en cada corrida.
- Recompilar el PDF después de cambiar `docs/paper.md`.
- Correr pruebas antes de hacer commit.

## Pendientes importantes

Estos puntos no bloquean continuar, pero conviene atenderlos en futuras iteraciones:

1. Versionar un manifiesto del corpus piloto con hashes, fuente, licencia, filtros y fecha, sin exponer datos privados.
2. Guardar artefactos de corrida final: `reports/eval_results.json`, predicciones crudas y matrices de confusión.
3. Congelar el entorno con `constraints.txt` o `pip freeze`.
4. Ejecutar notebooks con salidas limpias si se van a entregar como evidencia visual.
5. Implementar validación y early stopping real para BETO.
6. Ejecutar múltiples semillas y reportar desviación estándar o intervalos bootstrap.
7. Agregar prueba de significancia entre modelos.
8. Cachear respuestas LLM con modelo, fecha, prompt, tokens, latencia y costo.
9. Crear referencia manual producto-aspecto para auditar MAE, RMSE y Pearson de reputación.
10. Agregar GitHub Actions para correr pruebas automáticamente.

## Punto de revisión sobre citas

El paper compila y todas las entradas de `docs/references.bib` están citadas. La última revisión local detectó:

- 17 páginas en `docs/paper.pdf`;
- 29 fuentes únicas citadas;
- 113 marcadores de cita en el Markdown.

Si el profesor interpreta "2 a 3 citas por página" como fuentes únicas por página, la densidad queda cerca pero por debajo del rango. Si lo interpreta como marcadores de cita en texto, queda por arriba. Antes de la entrega final conviene decidir el criterio y, si hace falta, ajustar extensión o distribución de citas.

## Checklist antes de seguir

```bash
git status --short --branch
python3 -m pytest -q
python3 -m compileall src scripts tests
git diff --check
cd docs && pandoc paper.md --pdf-engine=typst --bibliography=references.bib --citeproc --toc --number-sections -o paper.pdf
```

Si todo pasa, hacer commit y push.
