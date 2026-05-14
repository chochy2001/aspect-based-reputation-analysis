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
- **Fuente:** https://huggingface.co/datasets/amazon_reviews_multi
- **Idioma:** Filtrar por `language="es"`
- **Tamaño:** la configuración `es` reporta 200K reseñas de entrenamiento, 5K de validación y 5K de prueba.
- **Campos originales relevantes:** `review_id`, `product_id`, `product_category`, `review_body`, `stars`, `language`.
- **Formato esperado tras normalización:** CSV con columnas `review_id`, `product_category`, `text`, `rating`.

```python
from datasets import load_dataset
ds = load_dataset("amazon_reviews_multi", "es")
train = ds["train"].to_pandas()
train = train.rename(columns={"review_body": "text", "stars": "rating"})
train[["review_id", "product_id", "product_category", "text", "rating"]].to_csv(
    "data/processed/reviews_train.csv", index=False
)
```

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

Si no existen columnas de anotación, el código genera pseudo-etiquetas con el lexicón para probar el flujo. No usar esas métricas como verdad terreno en el reporte final.

## Privacidad y uso de LLM

Los notebooks y scripts pueden enviar texto a Anthropic u OpenAI si se define una llave de API. Antes de ejecutar el enfoque LLM con datos reales, revisar consentimiento, términos de uso, anonimización y costo. No versionar reseñas privadas ni credenciales.

## Nota sobre recolección de datos

No se incluyen scrapers en el repositorio. Cualquier recolección directa desde plataformas debe respetar sus términos de servicio, robots.txt, límites de tasa y reglas de privacidad. Para una entrega reproducible se recomienda priorizar datasets con licencia explícita.

## Sample data

`sample/reviews_sample.csv` contiene 25 reseñas ficticias en español pensadas
para pruebas de humo del flujo (no es un dataset real).
