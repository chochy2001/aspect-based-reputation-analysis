# Estado del Proyecto: aspect-based-reputation-analysis

> Documento de estado y continuidad. Última actualización: 14 de mayo de 2026.
> Repositorio: <https://github.com/chochy2001/aspect-based-reputation-analysis>
> Rama activa: `main` (todo integrado, sin ramas pendientes).

---

## 1. Resumen ejecutivo

Proyecto final de la materia *Análisis y Procesamiento Inteligente de Textos* (APIT), UNAM Facultad de Ingeniería, semestre 2026-2. Tema: análisis de reputación basado en aspectos sobre reseñas de productos en español, comparando tres familias de métodos (lexicón+SVM, BETO afinado, LLM con few-shot prompting) y agregando predicciones por aspecto a puntuaciones de reputación 0 a 5.

La entrega consiste en:

1. Reporte académico de 34 páginas en `docs/paper.pdf`, compilado desde `docs/paper.md` con `pandoc + typst`.
2. Código fuente reproducible en `src/`, con pruebas automatizadas en `tests/`.
3. Scripts CLI para entrenar y evaluar cada enfoque.
4. Cinco notebooks didácticos en `notebooks/`.
5. Bibliografía verificada de 29 referencias en `docs/references.bib`.

---

## 2. Cumplimiento de la rúbrica del profesor

| Requerimiento del profesor | Ubicación en el paper | Estado |
|---|---|:-:|
| Título | YAML frontmatter | ✅ |
| Autores (Institución) | YAML `author` + `institute` | ✅ |
| Fecha | YAML `date` | ✅ |
| Abstract | `# Resumen (Abstract)` | ✅ |
| Palabras clave | Final del abstract | ✅ |
| Introducción | `# 1. Introducción` | ✅ |
| Antecedentes anecdóticos | `## 1.1` | ✅ |
| Motivación Social | `## 2.1` | ✅ |
| Motivación Académica | `## 2.2` | ✅ |
| Hipótesis | `## 2.3` (H₀, H₁, H₂ con umbrales) | ✅ |
| Descripción de la estructura | `## 2.4` | ✅ |
| Marco Teórico (teoría base, definiciones) | `# 3` (seis subsecciones) | ✅ |
| Aparato crítico (los otros perdedores) | `# 4` (seis subsecciones) | ✅ |
| Setup experimental, Hardware | `## 5.1` | ✅ |
| Setup experimental, Software | `## 5.2` | ✅ |
| Setup experimental, Objetivo y cómo evaluar | `## 5.3` | ✅ |
| Experimento n (material, evaluación, resultado) | `## 6.1` a `## 6.4` (cuatro experimentos con las tres partes cada uno) | ✅ |
| Resultado general (responde la hipótesis) | `# 7` (responde H₀, H₁, H₂) | ✅ |
| Análisis de resultados (choro chido) | `# 8` (seis subsecciones, extenso) | ✅ |
| Discusión y trabajo futuro | `# 9` | ✅ |
| Bibliografía (todo lo citado) | Final del paper, 29 entradas | ✅ |
| Promedio 2 a 3 citas por página | 88 inserciones / 34 páginas = 2.59 citas/pp | ✅ |

**Resultado:** 16 de 16 elementos del rubric cumplidos.

---

## 3. Verificaciones automáticas en verde

```
✅ Working tree limpio
✅ Branch local main == origin/main (commit f5e862f)
✅ pytest tests/test_smoke.py → 17/17 pasan
✅ pandoc + typst → docs/paper.pdf compila sin warnings críticos
✅ Todas las 29 claves BibTeX se citan al menos una vez, 0 huérfanas
✅ 0 em-dashes y 0 en-dashes en docs/ y README
✅ Sin firmas de IA en commits (autor = chochy2001 <ohchochy@gmail.com>)
✅ Sin marcadores prohibidos como "validando la hipótesis" o "se valida"
✅ Sin Co-Authored-By, Claude Code, AI-generated en ningún archivo
```

Comandos para reverificar en cualquier momento:

```bash
cd /Users/jorge/Documents/Escuela/APIT

# 1. Tests
python -m pytest tests/ -q

# 2. Recompilar el PDF
cd docs && pandoc paper.md \
  --pdf-engine=typst \
  --bibliography=references.bib \
  --citeproc \
  --toc \
  -o paper.pdf && cd ..

# 3. Densidad de citas
python -c "
import re
paper = open('docs/paper.md').read()
data = open('docs/paper.pdf','rb').read()
pages = len(re.findall(rb'/Type\s*/Page[^s]', data))
ins = paper.count('[@')
print(f'{ins} inserciones / {pages} pp = {ins/pages:.2f} citas/pp')
"

# 4. Tests CLI documentales (anti AI-marker)
python -m pytest tests/test_cli_and_docs.py -q
```

---

## 4. Estructura completa del repositorio

```
APIT/
├── README.md                        # Instalación, validación y reproducción
├── LICENSE                          # MIT
├── PROJECT_STATUS.md                # Este documento
├── pyproject.toml                   # Empaque y pytest config
├── requirements.txt                 # Dependencias de runtime
├── .gitignore                       # Ignora venvs, modelos, reports, PDF
├── docs/
│   ├── paper.md                     # Reporte académico (fuente)
│   ├── paper.pdf                    # PDF compilado (no versionado)
│   ├── metadata.yaml                # Frontmatter alterno para pandoc
│   ├── references.bib               # 29 referencias verificadas
│   └── reproducibility_checklist.md # Lista de auditoría antes de reportar
├── src/
│   ├── data/{loader,builder,preprocessing}.py
│   ├── aspects/extractor.py         # spaCy + fallback por keywords
│   ├── classical/{lexicon,svm_classifier}.py
│   ├── transformers_models/beto.py  # BETO con auxiliary sentence
│   ├── llm/prompting.py             # Anthropic u OpenAI con retry
│   ├── reputation/aggregator.py     # Agregación 0-5 con suavizado
│   └── evaluation/metrics.py        # F1, MAE, RMSE, Pearson
├── scripts/
│   ├── train_svm.py                 # CLI baseline clásico
│   ├── train_beto.py                # CLI fine-tuning transformer
│   └── eval_all.py                  # CLI evaluación comparativa
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_baseline_clasico.ipynb
│   ├── 03_beto_finetuning.ipynb
│   ├── 04_llm_prompting.ipynb
│   └── 05_comparativa_resultados.ipynb
├── tests/
│   ├── test_smoke.py                # Pruebas de humo sobre módulos puros
│   ├── test_core.py
│   ├── test_data_validation.py
│   └── test_cli_and_docs.py         # Anti AI-marker, frases prohibidas
└── data/
    ├── README.md
    └── sample/reviews_sample.csv    # 25 reseñas ficticias para CI
```

---

## 5. Qué está listo y verificado

### 5.1 Paper académico
- 34 páginas en PDF, 88 citas en texto, 2.59 citas por página.
- Bibliografía verificada con WebSearch y WebFetch, sin DOIs rotos.
- Atribuciones factuales corregidas tras revisión bibliográfica:
  - Sun et al. 2019: descripción exacta de la oración auxiliar QA-M/NLI-M.
  - Zhang et al. 2024: autor corregido `Deng, Yue`.
- Tablas 1 a 5 con macros aritméticamente consistentes.
- Hipótesis con umbrales numéricos verificables y respuesta cauta para cada una.
- Sección 5.4 de consideraciones éticas y legales sobre scraping, PII y contaminación LLM.

### 5.2 Código fuente
- 13 módulos Python compilan sin errores.
- Tres pipelines independientes:
  - Lexicón ampliado con coloquialismos mexicanos + SVM lineal TF-IDF.
  - BETO `bert-base-spanish-wwm-uncased` con auxiliary sentence y `truncation="only_first"`.
  - LLM con few-shot prompting, salida JSON, reintentos exponenciales (`tenacity`).
- Agregación reputacional bayesiana con suavizado `N/(N+k)`.
- Fallback graceful sin spaCy, sin GPU, sin API keys.
- Validaciones defensivas en cargadores de datos y dataset de BETO.

### 5.3 Reproducibilidad
- `pip install -e .` funciona (versión de `pysentimiento` corregida a `>=0.7.3`).
- `pytest` corre 17/17 pruebas de humo en menos de 100 ms.
- Scripts CLI con `--allow-pseudo-smoke` para evitar reportar métricas ficticias como verdad terreno.
- `docs/reproducibility_checklist.md` con la lista de evidencias exigibles para una corrida reportable.

### 5.4 Cumplimiento de políticas internas
- Sin atribución a IA en ningún archivo, ni en commits.
- Sin em-dashes ni en-dashes en documentación.
- Sin frases prohibidas detectadas por el test `test_documentation_avoids_known_delivery_risks`.

---

## 6. Qué falta para llegar a un trabajo publicable (techo metodológico)

Lo siguiente NO es necesario para la entrega de la materia, pero sí lo sería para convertir el piloto en un artículo de conferencia o un proyecto industrial:

### Estadística y validez
- Correr cada experimento con cinco semillas (42, 7, 13, 21, 100) y reportar media ± desviación estándar.
- Intervalos de confianza bootstrap al 95 % sobre F1, MAE y Pearson.
- Prueba de McNemar para diferencias BETO vs LLM y BETO vs lexicón.
- Kappa cuadrático ponderado (QWK) además de MAE para reputación, dado que la escala 0-5 es ordinal.
- Reportar Spearman además de Pearson, dado el rango acotado de los scores.

### Diagnóstico
- Matriz de confusión 3 × 3 por aspecto y modelo.
- Reliability diagram y Expected Calibration Error (ECE) para sustentar la afirmación de "mejor calibración" del softmax de BETO.
- Curvas de pérdida en validación de BETO, con `early stopping` real sobre F1 macro.
- Distribución de MAE por producto en lugar de solo el promedio.

### Datos
- Publicar el corpus anotado (o sus hashes y script de scrapeo reproducible), conforme a `docs/reproducibility_checklist.md`.
- Cobertura cross-dialectal: probar el modelo en reseñas de Argentina, Colombia o España.
- Cobertura multidominio: replicar el experimento en reseñas de servicios o productos financieros para validar generalización.

### Modelos alternativos
- Comparar BETO con MarIA y XLM-RoBERTa en el mismo split.
- Ablación de la técnica de auxiliary sentence: QA-M vs QA-B vs NLI-M vs NLI-B vs concatenación simple.
- Probar self-consistency para el LLM (mayoría sobre 5 muestreos con temperatura > 0).

### Despliegue
- Modelo empaquetado en `models/beto/` con `classifier_config.json` (ya soportado por el código).
- Servicio FastAPI o gRPC alrededor del clasificador.
- Dashboard de reputación por producto sobre Streamlit o Plotly Dash.

---

## 7. Cómo continuar en una sesión futura

### 7.1 Refrescar el contexto

```bash
cd /Users/jorge/Documents/Escuela/APIT
git status
git log --oneline -10
cat docs/project_status.md   # este documento
```

### 7.2 Antes de tocar el paper, validar que el build sigue verde

```bash
python -m pytest tests/ -q
cd docs && pandoc paper.md --pdf-engine=typst --bibliography=references.bib --citeproc --toc -o paper.pdf && cd ..
```

### 7.3 Si se quiere subir el rigor estadístico (lo más valioso a futuro)

1. Conseguir o generar un corpus anotado real con las columnas exigidas por `data/README.md`.
2. Implementar la rutina de cinco semillas en `scripts/train_svm.py` y `scripts/train_beto.py`.
3. Añadir un módulo `src/evaluation/significance.py` con McNemar y bootstrap.
4. Actualizar las tablas 1 a 4 con media y desviación estándar.
5. Cambiar el lenguaje del paper de "queda respaldada" a "se acepta con `p < 0.05` por bootstrap" donde aplique.

### 7.4 Reglas operativas heredadas

- Sin atribución a IA en ninguna parte del proyecto, ni en commits.
- Sin em-dashes ni en-dashes en documentación; usar comas o paréntesis.
- Mantener la densidad de citas en el rango 2.0 a 3.5 por página.
- No ejecutar `git push --force` sobre `main`.
- Antes de claim de éxito, correr `pytest` y recompilar el PDF.

---

## 8. Histórico de commits relevantes

```
f5e862f Integra fixes de 5 agentes en paralelo (BBVA, PDF, rigor, bib, repro)
c0baf7f Alinea README con paper: corpus piloto se distribuye bajo solicitud
fb1307b Alinea paper.md exacto al rubric del profesor
b28aac1 Documenta el comando real de compilación a PDF con typst
7e2c99b Limpieza de guiones em/en y compilación del PDF académico
6a119d1 Refuerzo de validaciones, robustez y correcciones del paper
87b4a46 Estructura inicial del proyecto: paper, código fuente y notebooks
```

---

## 9. Auditorías realizadas

El proyecto ha pasado por cinco auditorías independientes con calificaciones tras aplicar fixes:

| Eje auditado | Calificación inicial | Calificación tras fixes |
|---|:-:|:-:|
| Revisor BBVA hostil (defensa) | 7.5 / 10 | ~8.5 / 10 |
| Calidad tipográfica del PDF | 7 / 10 | 9 / 10 |
| Rigor metodológico y estadístico | 5 / 10 | 7.5 / 10 |
| Bibliografía verificada | 8.5 / 10 | 9.5 / 10 |
| Reproducibilidad real (install + run) | 6.5 / 10 | 8.5 / 10 |

Promedio actual estimado: **8.6 / 10** para entrega de licenciatura UNAM-FI. El techo de 10 sólido requiere los pendientes documentados en la Sección 6.

---

## 10. Contactos del equipo

- Gómez Vázquez Juan Pablo
- Martínez Miranda Juan Carlos
- Salgado Miranda Jorge
- Profesor titular: M. en C. Octavio Augusto Sánchez Velázquez
- GitHub del equipo: `chochy2001`
