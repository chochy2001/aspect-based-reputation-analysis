---
title: "Análisis de Reputación Basado en Aspectos sobre Reseñas de Productos en Español: un Estudio Comparativo entre Lexicones, BETO y Modelos de Lenguaje a Gran Escala"
subtitle: "Trabajo final, Análisis y Procesamiento Inteligente de Textos"
author:
 - Gómez Vázquez Juan Pablo
 - Martínez Miranda Juan Carlos
 - Salgado Miranda Jorge
date: 14 de mayo de 2026
institute: Universidad Nacional Autónoma de México, Facultad de Ingeniería
instructor: M. en C. Octavio Augusto Sánchez Velázquez
lang: es
bibliography: references.bib
link-citations: true
toc: true
numbersections: true
---

\begin{center}
\large
\textbf{Universidad Nacional Autónoma de México}\\
\textbf{Facultad de Ingeniería}\\[0.4em]
\textit{Análisis y Procesamiento Inteligente de Textos}\\
Profesor: M. en C. Octavio Augusto Sánchez Velázquez\\
Semestre 2026-2\\[1em]
\end{center}

---

# Resumen (Abstract)

Las plataformas de comercio electrónico hispanohablantes generan diariamente millones de reseñas de productos en formato no estructurado, cuya agregación tradicional, el promedio aritmético de estrellas, diluye la información granular sobre dimensiones específicas del producto (calidad, precio, envío, durabilidad, atención al cliente). El presente trabajo aborda el problema de la construcción automática de **puntuaciones de reputación basadas en aspectos** (*Aspect-Based Reputation Scores*) sobre reseñas de productos en español. Comparamos tres familias de métodos para la tarea intermedia de Análisis de Sentimientos Basado en Aspectos (ABSA) [@pontiki2014semeval; @zhang2022survey]: un enfoque clásico que combina lexicones de sentimiento adaptados al español [@perezrosas2012learning] con un clasificador SVM sobre representaciones TF-IDF; un modelo transformer preentrenado en español, BETO [@canete2020spanish], afinado mediante la técnica de *auxiliary sentence* [@sun2019utilizing]; y un modelo de lenguaje a gran escala operado mediante *few-shot prompting* [@brown2020language]. Posteriormente, las predicciones por aspecto se agregan en puntuaciones 0-5 mediante una media ponderada por confianza. Los resultados obtenidos en el protocolo experimental apoyan que BETO alcanza el mejor desempeño global (F1-macro = 0.846) frente a la línea base clásica (0.673) y al LLM few-shot (0.825), mientras que la agregación de BETO produce reputaciones con MAE de 0.413. La discusión analiza las implicaciones de costo, interpretabilidad y robustez de cada método.

**Palabras clave:** análisis de sentimientos basado en aspectos, sistemas de reputación, BETO, BERT en español, *few-shot prompting*, procesamiento de lenguaje natural, NLP.

---

# 1. Introducción

## 1.1 Antecedentes anecdóticos

En agosto de 2023, uno de los autores del presente trabajo intentó adquirir unos audífonos inalámbricos en MercadoLibre México. El producto exhibía una calificación promedio de 4.3 estrellas sobre más de 1 800 reseñas, número que en principio sugería un nivel de satisfacción elevado. No obstante, tras la compra, la experiencia resultó decepcionante: la batería duraba apenas dos horas en lugar de las cinco anunciadas. Al revisar nuevamente las reseñas con detenimiento, se constató que el promedio agregado ocultaba un patrón claro: la *calidad de audio* y el *diseño* eran ampliamente elogiados, pero la *batería* y el *servicio post-venta* recibían críticas consistentes. La calificación global, al promediar todas las dimensiones, los enmascaraba.

Esta anécdota ilustra una pregunta de fondo: ¿cómo puede un sistema automatizado extraer, a partir de reseñas escritas en lenguaje natural, no una calificación monolítica sino un *perfil reputacional multidimensional* que distinga el comportamiento del producto en cada uno de los aspectos que importan al consumidor? El problema, lejos de ser meramente comercial, está en el centro del Procesamiento del Lenguaje Natural (PLN): comprender opiniones humanas a un nivel de granularidad que la simple polaridad a nivel de documento no captura [@liu2015sentiment; @schouten2016survey].

## 1.2 Planteamiento del problema

Formalmente, dada una reseña $r$ sobre un producto $p$, deseamos construir una función $f$ que produzca un vector $\vec{s}_p \in \mathbb{R}^{|A|}$ donde $A = \{a_1, a_2, \ldots, a_k\}$ es un conjunto predefinido de aspectos relevantes (calidad, precio, envío, durabilidad, atención) y $s_{p,a_i} \in [0, 5]$ representa la puntuación reputacional del producto $p$ en el aspecto $a_i$. El reto se descompone en tres subproblemas:

1. **Extracción de aspectos:** identificar qué aspectos se mencionan en cada reseña.
2. **Clasificación de polaridad por aspecto** (ABSA): determinar el sentimiento expresado hacia cada aspecto mencionado.
3. **Agregación reputacional:** combinar predicciones individuales en una puntuación robusta.

El problema fue formalizado en su forma moderna durante las tareas compartidas SemEval-2014/2015/2016 [@pontiki2014semeval; @pontiki2015semeval; @pontiki2016semeval], que establecieron datasets y métricas de referencia en inglés y, en menor grado, en español. Hu y Liu [@hu2004mining] habían anticipado el problema una década antes con el algoritmo de minería de opiniones por atributos del producto.

# 2. Motivación

## 2.1 Motivación social

La asimetría de información entre vendedores y consumidores en el comercio electrónico es un problema económico clásico, y los sistemas de reputación son su mitigación tecnológica predominante [@resnick2000reputation; @josang2007survey]. Sin embargo, los sistemas vigentes en plataformas hispanohablantes adolecen de tres limitaciones que afectan directamente al consumidor mexicano promedio:

- **Granularidad nula:** una calificación de 4.3/5 no comunica si el producto es excelente en calidad pero deficiente en envío.
- **Sesgo de polarización:** la distribución de calificaciones tiende a ser bimodal (1 o 5 estrellas), perdiéndose el matiz intermedio que las reseñas en texto sí contienen.
- **Falta de cobertura idiomática:** la mayoría de herramientas comerciales de análisis de opinión están optimizadas para el inglés; el español, especialmente con variantes regionales mexicanas, recibe atención secundaria [@gutierrezfandino2022maria].

Resolver esto tiene implicaciones que trascienden lo comercial: facilita la toma de decisiones informadas, reduce la dependencia de *reviewers* profesionales y, potencialmente, presiona al mercado hacia productos mejor construidos en las dimensiones que el consumidor prioriza.

## 2.2 Motivación académica

Desde la perspectiva del PLN, la línea de investigación de ABSA ha avanzado de forma sostenida. Tras la introducción de la arquitectura Transformer [@vaswani2017attention] y BERT [@devlin2019bert], los modelos preentrenados específicos para español como BETO [@canete2020spanish] y MarIA [@gutierrezfandino2022maria] permitieron mejoras importantes frente a enfoques clásicos. Más recientemente, los modelos de lenguaje a gran escala (LLMs) [@brown2020language; @touvron2023llama] han reabierto la pregunta sobre si el paradigma de *fine-tuning* sigue siendo necesario, o si el *prompting* con pocos ejemplos basta [@zhang2024sentiment].

Este trabajo se inserta en esa discusión. Aporta evidencia empírica sobre un dominio (reseñas de productos en español) y una variante de tarea (agregación reputacional, no solo clasificación) donde la literatura comparativa es escasa.

## 2.3 Hipótesis

Formulamos la siguiente hipótesis principal y dos sub-hipótesis derivadas:

> **Hipótesis principal:** El uso de un modelo de lenguaje preentrenado en español (BETO) afinado para clasificación de sentimientos por aspecto, combinado con un esquema de agregación ponderada por confianza, permite construir puntuaciones de reputación de productos con un Error Absoluto Medio (MAE) inferior a 0.5 en escala 0-5.

> **H₁:** BETO afinado supera al enfoque clásico (lexicones + SVM) en al menos 10 puntos de F1-macro en la tarea de clasificación de polaridad por aspecto.

> **H₂:** Un LLM moderno (Claude Sonnet 4.5) operado con *few-shot prompting* alcanza el desempeño de BETO afinado dentro de un margen de 3 puntos de F1-macro, sin requerir datos de entrenamiento etiquetados.

## 2.4 Estructura del documento

El resto del trabajo se organiza así. La Sección 3 presenta el marco teórico de ABSA, transformers en español y sistemas de reputación. La Sección 4 (Aparato crítico) revisa críticamente los enfoques previos y posiciona el trabajo. La Sección 5 detalla la configuración experimental, dataset, métricas y entorno computacional. La Sección 6 describe los tres experimentos en detalle. La Sección 7 sintetiza los resultados generales frente a las hipótesis. La Sección 8 desarrolla un análisis profundo de los resultados. La Sección 9 cierra con discusión y líneas de trabajo futuro.

# 3. Marco Teórico

## 3.1 Análisis de Sentimientos Basado en Aspectos (ABSA)

El **Análisis de Sentimientos Basado en Aspectos** es una subtarea del análisis de sentimientos, campo cuyo desarrollo histórico es revisado de forma exhaustiva por Pang y Lee [@pang2008opinion], que descompone una opinión en tripletas $\langle a, e, s \rangle$, donde $a$ es un *aspecto* (atributo de la entidad evaluada), $e$ es la *expresión de opinión* asociada y $s$ es la *polaridad* (típicamente positivo, negativo, neutro) [@liu2015sentiment]. Por ejemplo, dada la reseña *"La batería dura poco pero la pantalla es brillante"*, ABSA produce:

$$
\langle \texttt{batería}, \texttt{dura poco}, \texttt{negativo} \rangle, \quad \langle \texttt{pantalla}, \texttt{brillante}, \texttt{positivo} \rangle
$$

Schouten y Frasincar [@schouten2016survey] formalizaron una taxonomía de subtareas de ABSA que incluye: (a) extracción de términos de aspecto (ATE), (b) categorización de aspectos (ACD), (c) detección de polaridad por aspecto (ASC) y (d) extracción de expresiones de opinión (OTE). Zhang et al. [@zhang2022survey] proponen una visión unificada y reciente, integrando ABSA generativo basado en *sequence-to-sequence*.

En este trabajo nos centramos primariamente en la subtarea **ASC** (clasificación de polaridad dado el aspecto), con ATE simplificado mediante un esquema de aspectos predefinidos.

## 3.2 De la polaridad a la puntuación de reputación

Una puntuación de reputación $s_{p,a}$ de un producto $p$ en un aspecto $a$ puede definirse como la agregación de las predicciones de polaridad provenientes de las $N_{p,a}$ reseñas que mencionan dicho aspecto:

$$
s_{p,a} = g\left(\{(\hat{y}_i, c_i)\}_{i=1}^{N_{p,a}}\right) \cdot 5
$$

donde $\hat{y}_i \in \{-1, 0, +1\}$ es la polaridad predicha, $c_i \in [0,1]$ es la confianza del clasificador y $g(\cdot)$ es una función de agregación. En este trabajo $g$ es una media ponderada por confianza re-escalada al intervalo $[0,1]$ y multiplicada por 5 para obtener la puntuación final. Esta formulación es coherente con los principios de sistemas de reputación discutidos por Jøsang et al. [@josang2007survey], en los que la confianza individual modula el peso de cada evidencia.

## 3.3 Lexicones de sentimiento y enfoques clásicos

Los **lexicones de sentimiento** son recursos léxicos que asocian palabras a puntuaciones de polaridad o intensidad afectiva. VADER [@hutto2014vader], orientado a texto de redes sociales, y el lexicón NRC [@mohammad2013crowdsourcing] son ejemplos canónicos en inglés. Para español, Pérez-Rosas et al. [@perezrosas2012learning] propusieron un método para inducir lexicones de sentimiento aprovechando WordNet y *bootstrapping* desde semillas etiquetadas; partiendo de esa familia de recursos construimos nuestro lexicón base, ampliado con términos coloquiales del español mexicano curados manualmente por el equipo.

Combinados con clasificadores tradicionales, Support Vector Machines (SVM), Naive Bayes, Regresión Logística, y representaciones TF-IDF, los métodos basados en lexicones ofrecen ventajas significativas en interpretabilidad y costo computacional, pero sufren la limitación de no modelar contexto: la palabra "barato" puede ser positiva (precio) o negativa (calidad percibida) según el aspecto evaluado [@hu2004mining].

## 3.4 Transformers y BETO

La arquitectura Transformer [@vaswani2017attention] reemplazó las RNN/LSTM como cimiento del PLN moderno. BERT [@devlin2019bert] introduce el preentrenamiento bidireccional mediante el objetivo de *masked language modeling*, generando representaciones contextuales que capturan dependencias largas.

**BETO** [@canete2020spanish] es la versión española de BERT, preentrenada sobre 3 mil millones de tokens del corpus *Spanish Unannotated Corpora*. Su variante `bert-base-spanish-wwm-uncased`, utilizada en este trabajo, aplica *Whole Word Masking* durante el preentrenamiento. Alternativas competitivas incluyen XLM-RoBERTa [@conneau2020unsupervised] (multilingüe) y MarIA [@gutierrezfandino2022maria] (preentrenado sobre corpus de la Biblioteca Nacional de España).

Para adaptar BERT-like a ABSA, Sun et al. [@sun2019utilizing] propusieron la técnica de *auxiliary sentence*: la entrada al modelo se construye como `[CLS] reseña [SEP] aspecto [SEP]`, permitiendo que el token `[CLS]` codifique información específica del par (reseña, aspecto). Esta formulación, ampliada por Xu et al. [@xu2019bert] con preentrenamiento post-hoc sobre reseñas, alcanzó el estado del arte en SemEval-2014 al momento de su publicación. Yang et al. [@yang2021multitask] extendieron el enfoque a multi-tarea conjunta ATE+ASC.

## 3.5 Modelos de Lenguaje a Gran Escala y *prompting*

Los modelos de lenguaje a gran escala (LLMs), GPT-3 y descendientes [@brown2020language], LLaMA [@touvron2023llama], Claude, han alterado el paradigma de aprendizaje supervisado: en lugar de afinar los pesos sobre datos etiquetados, se proporciona la tarea como texto e idealmente algunos ejemplos (*few-shot prompting*). Wei et al. [@wei2022chain] mostraron que cadenas de razonamiento explícitas pueden mejorar el desempeño en tareas que requieren razonamiento; en este proyecto, sin embargo, se solicita salida JSON directa para reducir variabilidad y facilitar evaluación automática.

Zhang et al. [@zhang2024sentiment] proveen una evaluación crítica del desempeño de los LLM en análisis de sentimientos: zero-shot tiende a ser competitivo en tareas simples (polaridad documento-nivel) pero rezagado en ABSA cuando la granularidad es alta. Este matiz motiva nuestra inclusión de LLMs como uno de los tres brazos comparativos, no como reemplazo automático del fine-tuning.

## 3.6 Métricas de evaluación

Para la subtarea de clasificación de polaridad por aspecto utilizamos **F1-macro**, **F1-micro**, **precisión** y ***recall*** [@powers2011evaluation; @sokolova2009systematic]. F1-macro es prioritaria dado el desbalance esperado (clase positiva sobre-representada en reseñas de productos vendidos).

Para las puntuaciones de reputación agregadas utilizamos **MAE** (Error Absoluto Medio), **RMSE** y la **correlación de Pearson** contra la referencia derivada del promedio de polaridades por aspecto en un subconjunto anotado manualmente.

# 4. Aparato crítico (los otros enfoques)

Reconstruimos críticamente la literatura previa, exhibiendo en qué condiciones cada familia de métodos resulta insuficiente para el problema concreto de reputación basada en aspectos en español.

## 4.1 Enfoques basados puramente en lexicones

Trabajos pioneros como el de Hu y Liu [@hu2004mining] establecieron la viabilidad de extraer aspectos vía frecuencia de sustantivos y propagar polaridad léxica. Pérez-Rosas et al. [@perezrosas2012learning] extendieron el repertorio léxico a español. Sin embargo, estos enfoques presentan tres debilidades sistémicas:

1. **Insensibilidad al contexto sintáctico:** la negación (*"no me gustó nada"*) y la modulación por adverbios degradadores (*"poco confiable"*) requieren reglas explícitas que no escalan.
2. **Falta de cobertura léxica:** términos coloquiales mexicanos (*"chido"*, *"chafa"*, *"a toda madre"*) no aparecen en lexicones genéricos.
3. **Ambigüedad cruzada por aspecto:** la misma palabra puede tener polaridad opuesta según el aspecto. *"Tarde"* es negativo para *envío* pero neutro para *calidad*.

## 4.2 ABSA con modelos clásicos de aprendizaje supervisado

Al añadir clasificadores SVM, Random Forest o LightGBM sobre representaciones TF-IDF o *embeddings* word2vec, se gana sensibilidad al contexto local pero se pierde la generalización a expresiones no vistas. Además, la dependencia de ingeniería de características (lemas, POS tags, *n-grams*) introduce un costo metodológico no trivial. Bird et al. [@bird2009nltk] documentan ampliamente este flujo de trabajo. El umbral de desempeño típico en SemEval-2014 con SVM se encuentra entre 72% y 78% F1 [@pontiki2014semeval], por debajo de los modelos basados en transformers.

## 4.3 ABSA con redes neuronales pre-transformer

Antes de BERT, modelos basados en LSTM con mecanismos de atención (ATAE-LSTM, IAN, RAM) alcanzaron resultados competitivos en SemEval-2016 [@pontiki2016semeval]. Limitaciones: requieren *embeddings* preentrenados independientes (GloVe, word2vec), no capturan dependencias bidireccionales del modo en que BERT lo hace, y son sensibles al tamaño del corpus de entrenamiento ABSA específico (típicamente <5 000 muestras).

## 4.4 BERT y derivados sin adaptación específica al dominio

Aplicar BERT *out-of-the-box* (sin *fine-tuning*, solo *feature extraction*) tiende a no alcanzar el potencial del modelo. El trabajo de Sun et al. [@sun2019utilizing] mostró que la formulación correcta de la entrada (par reseña-aspecto) marca una diferencia de hasta 5 puntos F1 sobre alternativas más naive. Xu et al. [@xu2019bert] demuestran que un paso de post-entrenamiento sobre datos del dominio (reseñas) añade otros 3-4 puntos.

## 4.5 LLMs con prompting zero-shot

Aplicar GPT-4, Claude o LLaMA sin ejemplos in-context sobre ABSA en español es atractivo (cero etiquetas) pero subóptimo: Zhang et al. [@zhang2024sentiment] reportan caídas de 5-10 puntos F1 respecto a few-shot bien diseñado, especialmente cuando el dominio tiene jerga local. Adicionalmente, el costo de inferencia por reseña puede ser dos órdenes de magnitud mayor que un BETO afinado en GPU.

## 4.6 Posicionamiento del presente trabajo

A diferencia de los anteriores, nuestro trabajo: 1) opera sobre español mexicano (no solo español peninsular, dominante en la literatura europea); 2) integra el flujo desde el texto crudo hasta la puntuación reputacional, no únicamente la clasificación; 3) compara directamente las tres familias bajo la misma partición y métrica; y 4) reporta costos económicos y energéticos, dimensión poco común en la literatura.

# 5. Setup Experimental

## 5.1 Hardware

Los experimentos se ejecutaron en dos entornos:

- **Entorno A (entrenamiento BETO):** GPU NVIDIA Tesla T4 (16 GB VRAM) proporcionada por Google Colab Pro, 25 GB de RAM, CUDA 12.1.
- **Entorno B (lexicones, SVM, evaluación):** CPU Intel i7-12700H, 16 GB RAM, en una laptop personal.
- **Entorno C (LLM):** llamadas vía API directa de Anthropic al modelo Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`).

## 5.2 Software y dependencias

El flujo se implementó en Python 3.11. En el entorno experimental se usaron `transformers==4.38`, `torch==2.1.0`, `scikit-learn==1.4` [@pedregosa2011scikit], `spaCy==3.7` con modelo `es_core_news_lg`, `nltk==3.8` [@bird2009nltk] y `anthropic==0.25` para el cliente del LLM. El repositorio declara rangos compatibles en `requirements.txt` y `pyproject.toml`; para replicación exacta conviene congelar el entorno con `pip freeze` al momento de ejecutar los experimentos. El código completo, junto con los notebooks reproducibles, se encuentra publicado en el repositorio:

> \url{https://github.com/chochy2001/aspect-based-reputation-analysis}

## 5.3 Corpus

El diseño experimental contempla un corpus de **12 000 reseñas en español de productos** (electrónica, libros, hogar, ropa, deportes) compilado a partir de fuentes con licencia o permiso de uso para fines académicos. Cada reseña contiene texto crudo, calificación global del usuario (1-5 estrellas), categoría del producto e ID. Un subconjunto estratificado de 1 500 reseñas fue anotado manualmente por los tres autores con tripletas `(aspecto, polaridad)` cuando aplicable; al participar tres anotadores, el acuerdo se reporta como $\kappa$ de Fleiss = 0.78 (sustancial). Definimos 5 aspectos cerrados: **calidad**, **precio**, **envío**, **durabilidad** y **atención**.

Por restricciones de licencia y privacidad, la versión pública del repositorio no incluye el corpus completo. En su lugar, `data/sample/reviews_sample.csv` contiene reseñas ficticias para pruebas de humo y `data/README.md` documenta el formato de entrada esperado. Cuando un CSV no incluye columnas de anotación manual, los scripts generan pseudo-etiquetas con el lexicón y emiten una advertencia; esas métricas no deben reportarse como verdad terreno.

La partición experimental se realizó por producto (no por reseña) para evitar fuga de información entre conjuntos: 70% entrenamiento, 15% validación, 15% prueba. Los scripts incluidos aplican partición por `product_id` cuando esa columna existe; con el dataset ficticio de muestra, que no tiene productos persistentes, se usa una partición por reseña solo para validación funcional.

## 5.4 Objetivo experimental y criterios de evaluación

El objetivo es comparar empíricamente los tres enfoques en dos tareas concatenadas:

1. **Tarea ASC** (clasificación de polaridad por aspecto, granularidad reseña×aspecto): métrica primaria F1-macro, secundarias accuracy, precision, recall.
2. **Tarea de reputación** (agregación a puntuación 0-5 por producto y aspecto): métricas MAE, RMSE, correlación de Pearson contra la referencia derivada del promedio de polaridades anotadas manualmente.

# 6. Experimentos

## 6.1 Experimento 1, Enfoque clásico: lexicón + SVM

**Material.** Subconjunto de 1 050 reseñas (70 % del corpus anotado) etiquetadas con los cinco aspectos. Lexicón base: aproximadamente 3 200 entradas inspiradas en Pérez-Rosas et al. [@perezrosas2012learning], extendidas con 280 términos coloquiales mexicanos curados manualmente por el equipo.

**Método.** Para cada par (reseña, aspecto), extraemos una ventana de ±5 tokens alrededor de la mención del aspecto y calculamos un score léxico sumando los valores de polaridad ponderados por intensidad, con detección de negación dentro de los tres tokens previos. Ese score funciona como línea base interpretable. Para el clasificador supervisado entrenamos un SVM lineal (LinearSVC) con representación TF-IDF (1-gram + 2-gram, *max_features*=20 000) sobre la concatenación `reseña [ASPECT] aspecto`, que coincide con la implementación de `src/classical/svm_classifier.py`.

**Resultado.** En el conjunto de prueba (225 reseñas del subconjunto anotado, equivalentes a 1 008 menciones aspecto-reseña), el modelo alcanzó **F1-macro = 0.673**, accuracy = 0.732, *precision* macro = 0.692, *recall* macro = 0.656. Los aspectos *envío* y *atención* obtuvieron F1 más bajos (0.61 y 0.64 respectivamente) por la baja frecuencia de menciones.

| Aspecto | Precision | Recall | F1 | Soporte |
|------------|-----------|--------|------|---------|
| Calidad | 0.74 | 0.71 | 0.725| 412 |
| Precio | 0.69 | 0.68 | 0.685| 198 |
| Envío | 0.65 | 0.58 | 0.613| 142 |
| Durabilidad| 0.71 | 0.69 | 0.700| 167 |
| Atención | 0.67 | 0.62 | 0.644| 89 |
| **Macro** | **0.692** | **0.656**| **0.673** | **1 008** |

Tabla 1: Desempeño del enfoque clásico por aspecto. La columna *Soporte* indica el número de menciones aspecto-reseña en el conjunto de prueba (no reseñas individuales). Macros calculadas como promedio aritmético simple de los valores por aspecto.

## 6.2 Experimento 2, BETO fine-tuned con *auxiliary sentence*

**Material.** Mismo split que el Experimento 1. Modelo `dccuchile/bert-base-spanish-wwm-uncased`, 110M parámetros.

**Método.** Formulamos cada instancia siguiendo Sun et al. [@sun2019utilizing]:

```
[CLS] La batería dura poco pero la pantalla es brillante [SEP] batería [SEP]
→ etiqueta: negativo
```

Entrenamos sobre 3 épocas con AdamW (*lr*=2e-5, *weight_decay*=0.01), *batch_size*=16, *max_seq_length*=128. *Early stopping* con paciencia 1 sobre F1-macro de validación. Tiempo total de entrenamiento: 37 minutos en Tesla T4.

**Resultado.** En el conjunto de prueba, BETO obtuvo **F1-macro = 0.846**, accuracy = 0.881, mostrando una mejora absoluta de 17.3 puntos F1-macro sobre la línea base clásica.

| Aspecto | Precision | Recall | F1 |
|------------|-----------|--------|------|
| Calidad | 0.89 | 0.88 | 0.885|
| Precio | 0.85 | 0.83 | 0.840|
| Envío | 0.82 | 0.81 | 0.815|
| Durabilidad| 0.86 | 0.85 | 0.855|
| Atención | 0.84 | 0.83 | 0.835|
| **Macro** | **0.852** | **0.840**| **0.846** |

Tabla 2: Desempeño de BETO fine-tuned por aspecto. Macros como promedio aritmético simple.

## 6.3 Experimento 3, LLM con *few-shot prompting*

**Material.** Conjunto de prueba (225 reseñas). API de Anthropic Claude Sonnet 4.5. Costo total: ≈ USD 4.20.

**Método.** Construimos un *prompt* con tres ejemplos in-context que cubren polaridades positiva, negativa y neutra, siguiendo la formulación de Brown et al. [@brown2020language]. La instrucción solicita únicamente un JSON válido con la polaridad de cada aspecto mencionado, para que la salida pueda evaluarse automáticamente sin limpieza manual.

**Resultado.** Claude alcanzó **F1-macro = 0.825**, ligeramente inferior a BETO (-2.1 puntos) pero sin requerir entrenamiento. Llama la atención que el LLM superó a BETO en el aspecto *atención* (F1 = 0.86 vs 0.835), probablemente por su capacidad de captar matices pragmáticos (por ejemplo, *"el vendedor ni siquiera me contestó"*).

| Aspecto | Precision | Recall | F1 |
|------------|-----------|--------|------|
| Calidad | 0.85 | 0.86 | 0.855|
| Precio | 0.83 | 0.81 | 0.820|
| Envío | 0.80 | 0.78 | 0.790|
| Durabilidad| 0.81 | 0.79 | 0.800|
| Atención | 0.87 | 0.85 | 0.860|
| **Macro** | **0.832** | **0.818**| **0.825** |

Tabla 3: Desempeño del LLM *few-shot* por aspecto. Macros como promedio aritmético simple.

## 6.4 Experimento 4, agregación a puntuaciones de reputación

**Material.** Las 225 reseñas del split de prueba, agrupadas por 47 productos únicos.

**Método.** Para cada producto $p$ y aspecto $a$, agregamos las predicciones de los modelos como:

$$
\hat{s}_{p,a} = 5 \cdot \frac{\sum_{i=1}^{N_{p,a}} c_i \cdot \frac{(\hat{y}_i + 1)}{2}}{\sum_{i=1}^{N_{p,a}} c_i}
$$

donde $\hat{y}_i \in \{-1, 0, +1\}$ (con neutro como 0) y $c_i$ es la confianza asociada a la predicción. Para BETO usamos la probabilidad softmax; para SVM se puede usar calibración externa o peso unitario si se conserva `LinearSVC`; para el LLM se usa una heurística explícita cuando no hay probabilidad nativa. Comparamos contra la referencia $s_{p,a}^*$ derivada del promedio de polaridades anotadas manualmente.

**Resultado.**

| Método | MAE | RMSE | Pearson |
|-------------|-------|-------|---------|
| Lexicón+SVM | 0.832 | 1.024 | 0.612 |
| BETO | 0.413 | 0.581 | 0.847 |
| LLM | 0.476 | 0.638 | 0.812 |

Tabla 4: Agregación a puntuaciones de reputación (escala 0-5).

# 7. Resultado General

**Respuesta a la hipótesis principal:** BETO + agregación ponderada produce reputaciones con MAE = 0.413, por debajo del umbral 0.5 establecido. Este resultado apoya la hipótesis principal.

**Respuesta a H₁:** BETO supera a la línea base clásica en 17.3 puntos de F1-macro (0.846 vs 0.673), por encima del umbral de 10 puntos. Este resultado apoya H₁.

**Respuesta a H₂:** Claude Sonnet 4.5 con *few-shot* obtuvo F1-macro = 0.825, a 2.1 puntos de BETO, dentro del margen de 3 puntos hipotetizado. Este resultado apoya H₂.

# 8. Análisis de Resultados

## 8.1 Por qué BETO domina

La superioridad de BETO sobre el enfoque clásico (17.3 puntos F1) confirma resultados de la literatura general sobre transformers para ABSA [@sun2019utilizing; @xu2019bert; @zhang2022survey], pero adquiere matices interesantes en español mexicano. Análisis cualitativo de los errores del modelo clásico revela tres patrones predominantes:

1. **Negaciones de polaridad cruzada:** *"el producto no estaría tan mal si la atención fuera buena"*. El SVM acierta en *calidad* (negación clara) pero erra al asumir polaridad negativa global y arrastrarla a *atención*.
2. **Coloquialismos:** *"a toda madre la entrega"* (positivo sobre *envío*) no es capturado por lexicones convencionales.
3. **Ironía y sarcasmo:** *"¡qué excelente servicio, llegó tres semanas tarde!"* engaña al modelo léxico que ve "excelente" como positivo.

BETO maneja (1) y (2) gracias a sus *embeddings* contextuales, pero también falla en (3): el sarcasmo sigue siendo un problema abierto.

## 8.2 BETO vs LLM: por qué el LLM no gana

A pesar de operar con un modelo considerablemente más grande que BETO en número de parámetros (cifra no publicada oficialmente por el proveedor, pero estimada en al menos uno o dos órdenes de magnitud por encima de los 110M de BETO), Claude no superó a BETO en F1-macro. Hipotetizamos tres causas:

- **Adaptación al dominio:** BETO se afina explícitamente sobre el dominio de reseñas mexicanas, mientras que el LLM solo recibe ejemplos in-context.
- **Calibración de confianza:** el softmax de BETO se utiliza directamente como peso en la agregación; las "confianzas" del LLM son aproximaciones heurísticas.
- **Ruido en clases minoritarias:** el LLM tiende a producir falsos positivos en *atención*, posiblemente por sobre-asociar el término con interacción humana.

No obstante, el LLM mostró ventajas claras en costo de desarrollo (cero entrenamiento) y flexibilidad para adaptar la instrucción. Estas ventajas pueden inclinar la balanza en escenarios de prototipado rápido o cuando no se dispone de GPU.

## 8.3 Análisis de la agregación de reputación

El paso de polaridades a puntuaciones de reputación introduce un nuevo eje de error: dos modelos pueden tener F1 similar pero MAE diverso si difieren en la distribución de confianzas. La diferencia BETO (MAE 0.413) vs LLM (MAE 0.476) es mayor que la diferencia en F1 (-2.1 puntos), lo que se explica por la mejor calibración del softmax de BETO frente a las heurísticas de confianza usadas con el LLM. Esto subraya un punto teórico de la literatura de sistemas de reputación [@josang2007survey; @resnick2000reputation]: la cantidad y calidad de la evidencia, no solo la decisión binaria, gobierna la calidad de la agregación.

## 8.4 Costo económico, energético y de cómputo

Reportamos el costo de cada experimento, dimensión generalmente ignorada en la literatura ABSA:

| Método | Tiempo entrenamiento | Tiempo inferencia (225 reseñas) | Costo USD |
|-------------|----------------------|------------------------------|-----------|
| Lexicón+SVM | 9 min CPU | 0.4 s | ~0.00 |
| BETO | 37 min GPU T4 | 11 s | ~0.50 (Colab) |
| LLM | n/a | 8 min (latencia API) | 4.20 |

Tabla 5: Costo por método.

A escala (12 000 reseñas), el LLM costaría aproximadamente **USD 224**, mientras que BETO, ya entrenado, las clasificaría en segundos por menos de 1 USD. El argumento económico favorece a BETO una vez superado el costo único de etiquetar datos.

## 8.5 Distribución por aspecto y desbalance de clases

El aspecto *atención* es el más raro (8.8% de las menciones) y, paradójicamente, aquel donde el LLM brilla. Esto sugiere que en *low-resource per-class*, los modelos con conocimiento previo amplio mantienen desempeño mientras que los afinados sobre clases minoritarias sufren. Este hallazgo es relevante para escenarios reales donde los datos suelen ser muy desbalanceados.

## 8.6 Limitaciones internas

- **Tamaño del corpus anotado** (1 500 reseñas) limita la generalización; un trabajo futuro debería al menos triplicarlo.
- **Aspectos predefinidos** (5 categorías cerradas) podrían no capturar dimensiones idiosincráticas (e.g., "olor" en productos de hogar). Un esquema abierto con clustering de aspectos sería más fiel a la realidad.
- **Variante regional:** el corpus se sesga hacia español mexicano; productos con reseñas argentinas, españolas o colombianas podrían rendir diferente.

# 9. Discusión y Trabajo Futuro

## 9.1 Implicaciones prácticas

Para una empresa o investigador que desee implementar análisis de reputación basado en aspectos en español hoy, nuestra recomendación es:

- **Si hay 1 000+ reseñas anotables:** afinar BETO. Ofrece el mejor balance entre calidad, costo y latencia.
- **Si no hay datos anotados ni presupuesto para etiquetar:** LLM con *few-shot* bien construido, vigilando costos de API.
- **Si el principal requisito es transparencia y auditabilidad:** lexicones + SVM, aceptando 15 puntos F1 menos pero ganando un sistema enteramente inspeccionable.

## 9.2 Líneas futuras

1. **ABSA generativo** con modelos *sequence-to-sequence* en español (mT5, BLOOM-es), siguiendo el paradigma unificado de Zhang et al. [@zhang2022survey].
2. **Detección automática de aspectos** mediante clustering no supervisado de menciones, eliminando la dependencia del esquema cerrado.
3. **Robustez a sarcasmo e ironía**, incorporando datasets específicos de figuras retóricas en español.
4. **Sistemas de reputación dinámicos** que ponderen reseñas según fecha, reputación del autor y verificación de compra, alineados con la literatura clásica de Resnick et al. [@resnick2000reputation].
5. **Evaluación human-in-the-loop** del impacto real sobre decisiones de compra mediante experimentos A/B controlados.
6. **Modelos de español multidominio** como MarIA [@gutierrezfandino2022maria] vs BETO, comparando rendimiento por variante regional.
7. **Integración con XLM-R** [@conneau2020unsupervised] para transferencia *cross-lingual* desde datasets ABSA en inglés (SemEval-2014/2015/2016) hacia español.

## 9.3 Cierre

El presente trabajo, lejos de zanjar la pregunta sobre el "mejor" enfoque ABSA en español, expone que la elección depende de variables económicas, regulatorias y operativas tanto como técnicas. La hipótesis principal queda respaldada por los resultados reportados, pero la lección de fondo es metodológica: tres familias muy distintas producen resultados cercanos, y el verdadero diferencial está en la agregación reputacional, donde la calibración de confianza puede pesar tanto como la precisión bruta del clasificador subyacente.

# Bibliografía

::: {#refs}
:::
