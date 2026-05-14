# Checklist de reproducibilidad

Usar esta lista antes de reportar resultados como finales.

## Datos

- Fuente exacta del dataset, URL o identificador.
- Licencia o permiso de uso.
- Fecha de descarga.
- Hash de archivos crudos y procesados.
- Script de normalización usado.
- Criterios de filtrado y muestreo.
- Evidencia de anonimización cuando aplique.

## Anotación

- Guía de anotación por aspecto y polaridad.
- Número de anotadores.
- Unidad anotada: reseña, par reseña-aspecto o tripleta.
- Acuerdo entre anotadores y método de cálculo.
- Criterio de desempate o adjudicación.
- Distribución por aspecto y etiqueta.

## Experimentos

- Versión de Python y sistema operativo.
- `pip freeze` o archivo de restricciones.
- Modelo spaCy usado.
- Revisión o commit del modelo de Hugging Face si se usa BETO.
- Semillas de partición y entrenamiento.
- Comandos exactos ejecutados.
- Archivos de salida en `reports/`.
- Predicciones crudas por modelo.
- Matrices de confusión y métricas por aspecto.

## LLM

- Proveedor y modelo exacto.
- Fecha de ejecución.
- Prompt completo.
- Parámetros de inferencia.
- Respuestas crudas.
- Tasa de JSON inválido.
- Tokens y costo.
- Confirmación de privacidad y términos de uso.
