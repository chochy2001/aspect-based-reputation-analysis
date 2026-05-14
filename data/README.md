# Datos

Esta carpeta contiene los datasets para entrenamiento y evaluación del proyecto APIT.

## Estructura

```
data/
├── README.md            # Este archivo
├── sample/              # Datos de ejemplo (ficticios) para pruebas rápidas
│   └── reviews_sample.csv
├── raw/                 # (No versionado) datasets crudos descargados
└── processed/           # (No versionado) datasets preprocesados
```

## Datasets recomendados y trazabilidad

### 1. Amazon Reviews (multilingual)
- **Fuente recomendada actual:** https://huggingface.co/datasets/mteb/amazon_reviews_multi
- **Fuente original/paper:** https://arxiv.org/abs/2010.02573
- **Idioma:** usar el subset `es`.
- **Tamaño:** la configuración `es` reporta 200K reseñas de entrenamiento, 5K de validación y 5K de prueba.
- **Campos originales relevantes:** dependen del espejo usado; en `mteb/amazon_reviews_multi` aparecen `id`, `text`, `label` y `label_text`.
- **Formato esperado tras normalización:** CSV con columnas `review_id`, `product_category`, `text`, `rating`.

```python
from datasets import load_dataset
ds = load_dataset("mteb/amazon_reviews_multi", "es")
train = ds["train"].to_pandas()
train = train.rename(columns={"id": "review_id", "label": "rating"})
train["rating"] = train["rating"] + 1  # labels 0-4 -> rating 1-5
train["product_category"] = "amazon_multi"
train[["review_id", "product_category", "text", "rating"]].to_csv(
    "data/processed/reviews_train.csv", index=False
)
```

Nota: el identificador histórico `amazon_reviews_multi` puede fallar según la versión de `datasets`; por eso se recomienda el espejo `mteb/amazon_reviews_multi` y registrar el commit/fecha usada.

### 2. MercadoLibre Reviews
- **Fuente:** https://www.kaggle.com/datasets/ (buscar "mercadolibre reviews")
- **Nota:** requiere cuenta de Kaggle, autenticación CLI y revisión de licencia del dataset específico.

### 3. MeOFFenDES / TASS Workshop
- **Fuente:** http://tass.sepln.org/
- **Uso:** Validación con datos académicos anotados de sentimiento en español.

## Formato esperado por los scripts

Todos los CSVs deben tener al menos estas columnas:

| Columna           | Tipo   | Descripción                                  |
|-------------------|--------|----------------------------------------------|
| review_id         | str    | Identificador único de la reseña             |
| product_category  | str    | Categoría del producto (electrónica, etc.)   |
| text              | str    | Texto de la reseña en español                |
| rating            | int    | Puntuación 1-5 dada por el usuario           |

Para evaluación experimental real, agregar anotaciones por aspecto:

| Columna aceptada                  | Tipo | Descripción |
|-----------------------------------|------|-------------|
| aspect o aspecto                  | str  | Aspecto mencionado. Se normaliza a calidad, precio, envío, durabilidad o atención cuando aplica. |
| label, sentiment, polarity o polaridad | str | Etiqueta `pos`, `neg` o `neu` (`positivo`, `negativo`, `neutro` también se aceptan). |

Si no existen columnas de anotación, el código solo permite pseudo-etiquetas con la bandera explícita `--allow-pseudo-smoke`. No usar esas métricas como verdad terreno en el reporte final.

## Trazabilidad mínima para resultados reportables

Antes de reportar F1, MAE o comparaciones entre modelos, guardar junto con el CSV:

- fuente exacta, URL o identificador del dataset;
- licencia o permiso de uso;
- fecha de descarga y filtros aplicados;
- script de normalización;
- hashes de archivos crudos y procesados;
- criterio de muestreo;
- guía de anotación y resolución de desacuerdos;
- partición train/test o train/valid/test usada;
- archivo `reports/eval_results.json` generado por `scripts/eval_all.py`.

Un ejemplo de archivo anotado puede tener una fila por par reseña-aspecto:

```csv
review_id,product_id,product_category,text,rating,aspect,label
r001,p001,electronica,"La batería dura mucho pero el precio es alto.",4,durabilidad,pos
r001,p001,electronica,"La batería dura mucho pero el precio es alto.",4,precio,neg
```

## Privacidad y uso de LLM

Los notebooks y scripts pueden enviar texto a Anthropic u OpenAI si se define una llave de API. Antes de ejecutar el enfoque LLM con datos reales, revisar consentimiento, términos de uso, anonimización y costo. No versionar reseñas privadas ni credenciales.

## Nota sobre recolección de datos

No se incluyen scrapers en el repositorio. Cualquier recolección directa desde plataformas debe respetar sus términos de servicio, robots.txt, límites de tasa y reglas de privacidad. Para una entrega reproducible se recomienda priorizar datasets con licencia explícita.

## Sample data

`sample/reviews_sample.csv` contiene 25 reseñas ficticias en español pensadas
para pruebas de humo del flujo (no es un dataset real).
