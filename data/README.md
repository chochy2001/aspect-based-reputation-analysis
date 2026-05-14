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

## Datasets recomendados

### 1. Amazon Reviews (multilingual)
- **Fuente:** https://huggingface.co/datasets/amazon_reviews_multi
- **Idioma:** Filtrar por `language="es"`
- **Tamaño:** ~210K reseñas en español
- **Formato esperado tras descarga:** CSV con columnas `review_id`, `product_category`, `text`, `rating`

```python
from datasets import load_dataset
ds = load_dataset("amazon_reviews_multi", "es")
```

### 2. MercadoLibre Reviews
- **Fuente:** https://www.kaggle.com/datasets/ (buscar "mercadolibre reviews")
- **Nota:** Requiere cuenta de Kaggle y autenticación CLI.

### 3. MeOFFenDES / TASS Workshop
- **Fuente:** http://tass.sepln.org/
- **Uso:** Validación con datos académicos anotados de sentimiento en español.

## Formato esperado

Todos los CSVs deben tener al menos estas columnas:

| Columna           | Tipo   | Descripción                                  |
|-------------------|--------|----------------------------------------------|
| review_id         | str    | Identificador único de la reseña             |
| product_category  | str    | Categoría del producto (electrónica, etc.)   |
| text              | str    | Texto de la reseña en español                |
| rating            | int    | Puntuación 1-5 dada por el usuario           |

## Sample data

`sample/reviews_sample.csv` contiene 25 reseñas ficticias en español pensadas
para test de humo del pipeline (no es un dataset real).
