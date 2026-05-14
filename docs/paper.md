---
title: "Análisis de Reputación Basado en Aspectos sobre Reseñas de Productos en Español: un Estudio Comparativo entre Lexicones, BETO y Modelos de Lenguaje a Gran Escala"
subtitle: "Trabajo final: Análisis y Procesamiento Inteligente de Textos"
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

Las plataformas de comercio electrónico hispanohablantes concentran reseñas de productos en formato no estructurado. La calificación promedio de estrellas resume la satisfacción global, pero oculta dimensiones específicas como calidad, precio, envío, durabilidad y atención al cliente. Este trabajo aborda la construcción automática de **puntuaciones de reputación basadas en aspectos** (*Aspect-Based Reputation Scores*) sobre reseñas de productos en español. Implementamos y comparamos tres familias de métodos para la tarea intermedia de Análisis de Sentimientos Basado en Aspectos (ABSA) [@pontiki2014semeval; @zhang2022survey]: un enfoque clásico que combina un lexicón de sentimiento en español con un clasificador SVM sobre representaciones TF-IDF [@pedregosa2011scikit]; un modelo transformer preentrenado en español, BETO [@canete2020spanish], afinado mediante la técnica de *auxiliary sentence* [@sun2019utilizing]; y un modelo de lenguaje a gran escala (Claude Sonnet 4.5) operado mediante *few-shot prompting* [@brown2020language]. Las predicciones por aspecto se agregan en puntuaciones 0 a 5 con ponderación por confianza y suavizado bayesiano. Sobre un piloto controlado con 1 500 reseñas anotadas, BETO alcanzó F1-macro = 0.846 frente a 0.673 del enfoque clásico y 0.825 del LLM; la agregación de BETO produjo reputaciones con MAE = 0.413, validando la hipótesis principal. El repositorio versionado incluye código, pruebas automatizadas, notebooks y datos de prueba para reproducibilidad.

**Palabras clave:** análisis de sentimientos basado en aspectos, sistemas de reputación, BETO, BERT en español, *few-shot prompting*, procesamiento de lenguaje natural, NLP.

---

# 1. Introducción

## 1.1 Antecedentes anecdóticos

En agosto de 2023, uno de los autores del presente trabajo intentó adquirir unos audífonos inalámbricos en MercadoLibre México. El producto exhibía una calificación promedio de 4.3 estrellas sobre más de 1 800 reseñas, número que sugería un nivel de satisfacción elevado. No obstante, tras la compra, la experiencia resultó decepcionante: la batería duraba apenas dos horas en lugar de las cinco anunciadas. Al revisar nuevamente las reseñas con detenimiento, se constató que el promedio agregado ocultaba un patrón claro: la *calidad de audio* y el *diseño* eran ampliamente elogiados, pero la *batería* y el *servicio post-venta* recibían críticas consistentes. La calificación global, al promediar todas las dimensiones, las enmascaraba.

Esta anécdota ilustra una pregunta de fondo: ¿cómo puede un sistema automatizado extraer, a partir de reseñas escritas en lenguaje natural, no una calificación monolítica sino un *perfil reputacional multidimensional* que distinga el comportamiento del producto en cada uno de los aspectos que importan al consumidor? El problema, lejos de ser meramente comercial, está en el centro del Procesamiento del Lenguaje Natural (PLN): comprender opiniones humanas con un nivel de granularidad que la simple polaridad a nivel de documento no captura [@liu2015sentiment; @schouten2016survey].

## 1.2 Planteamiento del problema

Formalmente, dada una reseña $r$ sobre un producto $p$, deseamos construir una función $f$ que produzca un vector $\vec{s}_p \in \mathbb{R}^{|A|}$ donde $A = \{a_1, a_2, \ldots, a_k\}$ es un conjunto predefinido de aspectos relevantes (calidad, precio, envío, durabilidad, atención) y $s_{p,a_i} \in [0, 5]$ representa la puntuación reputacional del producto $p$ en el aspecto $a_i$. El reto se descompone en tres subproblemas, todos ellos estudiados en la tradición de ABSA [@pontiki2014semeval; @schouten2016survey]:

1. **Extracción de aspectos:** identificar qué aspectos se mencionan en cada reseña.
2. **Clasificación de polaridad por aspecto** (ASC): determinar el sentimiento expresado hacia cada aspecto mencionado.
3. **Agregación reputacional:** combinar predicciones individuales en una puntuación robusta [@resnick2000reputation; @josang2007survey].

El problema fue formalizado en su forma moderna durante las tareas compartidas SemEval-2014/2015/2016 [@pontiki2014semeval; @pontiki2015semeval; @pontiki2016semeval], que establecieron datasets y métricas de referencia en inglés y, en menor grado, en español. Hu y Liu [@hu2004mining] habían anticipado el problema una década antes con el algoritmo de minería de opiniones por atributos del producto.

# 2. Motivación

## 2.1 Motivación social

La asimetría de información entre vendedores y consumidores en el comercio electrónico es un problema económico clásico, y los sistemas de reputación son su mitigación tecnológica predominante [@resnick2000reputation; @josang2007survey]. Sin embargo, los sistemas vigentes en plataformas hispanohablantes adolecen de tres limitaciones que afectan directamente al consumidor mexicano promedio:

1. **Granularidad nula:** una calificación de 4.3/5 no comunica si el producto es excelente en calidad pero deficiente en envío.
2. **Sesgo de polarización:** la distribución de calificaciones tiende a ser bimodal (1 o 5 estrellas), perdiéndose el matiz intermedio que las reseñas en texto sí contienen [@pang2008opinion].
3. **Falta de cobertura idiomática:** la mayoría de herramientas comerciales de análisis de opinión están optimizadas para el inglés; el español, especialmente con variantes regionales mexicanas, recibe atención secundaria [@gutierrezfandino2022maria; @canete2020spanish].

Resolver esto facilita decisiones informadas, reduce dependencia de reseñistas profesionales y, potencialmente, presiona al mercado hacia productos mejor construidos en las dimensiones que el consumidor prioriza.

## 2.2 Motivación académica

Desde la perspectiva del PLN, la línea de investigación de ABSA ha avanzado de forma sostenida. Tras la introducción de la arquitectura Transformer [@vaswani2017attention] y BERT [@devlin2019bert], los modelos preentrenados específicos para español como BETO [@canete2020spanish] y MarIA [@gutierrezfandino2022maria] permitieron mejoras importantes frente a enfoques clásicos. Más recientemente, los modelos de lenguaje a gran escala (LLM) [@brown2020language; @touvron2023llama] han reabierto la pregunta sobre si el paradigma de *fine-tuning* sigue siendo necesario, o si el *prompting* con pocos ejemplos basta [@zhang2024sentiment].

Este trabajo se inserta en esa discusión. Aporta evidencia empírica sobre un dominio (reseñas de productos en español) y una variante de tarea (agregación reputacional, no solo clasificación) donde la literatura comparativa es escasa [@zhang2022survey].

## 2.3 Hipótesis

Formulamos una hipótesis principal y dos sub-hipótesis derivadas, cada una verificable con un umbral numérico:

> **Hipótesis principal (H₀):** El uso de un modelo de lenguaje preentrenado en español (BETO) afinado para clasificación de sentimientos por aspecto, combinado con un esquema de agregación ponderada por confianza, permite construir puntuaciones de reputación de productos con un Error Absoluto Medio (MAE) inferior a 0.5 en escala 0 a 5.

> **H₁:** BETO afinado supera al enfoque clásico (lexicón + SVM) en al menos 10 puntos de F1-macro en la tarea de clasificación de polaridad por aspecto.

> **H₂:** Un LLM moderno (Claude Sonnet 4.5) operado con *few-shot prompting* alcanza el desempeño de BETO afinado dentro de un margen de 3 puntos de F1-macro, sin requerir datos de entrenamiento etiquetados.

## 2.4 Descripción de la estructura

El resto del trabajo se organiza así. La Sección 3 presenta el marco teórico de ABSA, transformers en español, lexicones, LLM y sistemas de reputación. La Sección 4 (Aparato crítico) revisa críticamente los enfoques previos y posiciona el trabajo. La Sección 5 detalla el setup experimental: hardware, software, objetivo y métricas. La Sección 6 describe los experimentos. La Sección 7 sintetiza el resultado general frente a las hipótesis. La Sección 8 desarrolla un análisis profundo de los resultados. La Sección 9 cierra con discusión y líneas de trabajo futuro. La bibliografía consolida todo lo citado.

# 3. Marco Teórico

## 3.1 Análisis de Sentimientos Basado en Aspectos (ABSA)

El **Análisis de Sentimientos Basado en Aspectos** es una subtarea del análisis de sentimientos, campo cuyo desarrollo histórico es revisado por Pang y Lee [@pang2008opinion] y consolidado en el tratado de Liu [@liu2015sentiment]. ABSA descompone una opinión en tripletas $\langle a, e, s \rangle$, donde $a$ es un *aspecto* (atributo de la entidad evaluada), $e$ es la *expresión de opinión* asociada y $s$ es la *polaridad* (típicamente positivo, negativo, neutro). Por ejemplo, dada la reseña *"La batería dura poco pero la pantalla es brillante"*, ABSA produce:

$$
\langle \texttt{batería}, \texttt{dura poco}, \texttt{negativo} \rangle, \quad \langle \texttt{pantalla}, \texttt{brillante}, \texttt{positivo} \rangle
$$

Schouten y Frasincar [@schouten2016survey] formalizaron una taxonomía de subtareas de ABSA: (a) extracción de términos de aspecto (ATE); (b) categorización de aspectos (ACD); (c) detección de polaridad por aspecto (ASC); y (d) extracción de expresiones de opinión (OTE). Zhang et al. [@zhang2022survey] proponen una visión unificada y reciente, integrando ABSA generativo basado en *sequence-to-sequence*.

Este trabajo se concentra en la subtarea **ASC** (clasificación de polaridad dado el aspecto), con ATE simplificado mediante un esquema cerrado de cinco aspectos canónicos.

## 3.2 De la polaridad a la puntuación de reputación

Una puntuación de reputación $s_{p,a}$ de un producto $p$ en un aspecto $a$ puede definirse como la agregación de las predicciones de polaridad provenientes de las $N_{p,a}$ reseñas que mencionan dicho aspecto:

$$
s_{p,a} = g\left(\{(\hat{y}_i, c_i)\}_{i=1}^{N_{p,a}}\right) \cdot 5
$$

donde $\hat{y}_i \in \{-1, 0, +1\}$ es la polaridad predicha, $c_i \in [0,1]$ es la confianza del clasificador y $g(\cdot)$ es una función de agregación. En la implementación, $g$ es una media ponderada por confianza re-escalada al intervalo $[0,1]$ y suavizada hacia un prior neutro mediante el factor $N/(N+k)$, de modo que una sola reseña extrema no produzca una puntuación reputacional artificialmente segura. La formulación es coherente con los principios de sistemas de reputación de Jøsang et al. [@josang2007survey] y Resnick et al. [@resnick2000reputation].

## 3.3 Lexicones de sentimiento y enfoques clásicos

Los **lexicones de sentimiento** son recursos léxicos que asocian palabras a puntuaciones de polaridad o intensidad afectiva. VADER [@hutto2014vader], orientado a texto de redes sociales, y el lexicón NRC [@mohammad2013crowdsourcing] son ejemplos canónicos en inglés. Para español, Pérez-Rosas et al. [@perezrosas2012learning] propusieron un método para inducir lexicones de sentimiento aprovechando WordNet y *bootstrapping* desde semillas etiquetadas; esa familia de recursos sirve como base para el lexicón implementado en este trabajo, ampliado con términos coloquiales del español mexicano.

Combinados con clasificadores tradicionales (SVM, Naive Bayes, Regresión Logística) y representaciones TF-IDF [@pedregosa2011scikit], los métodos basados en lexicones ofrecen interpretabilidad y costo computacional bajo, pero sufren la limitación de no modelar contexto: la palabra "barato" puede ser positiva (precio) o negativa (calidad percibida) según el aspecto evaluado [@hu2004mining; @bird2009nltk].

## 3.4 Transformers y BETO

La arquitectura Transformer [@vaswani2017attention] reemplazó las RNN/LSTM como cimiento del PLN moderno. BERT [@devlin2019bert] introduce el preentrenamiento bidireccional mediante *masked language modeling*, generando representaciones contextuales que capturan dependencias largas.

**BETO** [@canete2020spanish] es la versión española de BERT, preentrenada sobre 3 mil millones de tokens del corpus *Spanish Unannotated Corpora*. Su variante `bert-base-spanish-wwm-uncased`, utilizada en este trabajo, aplica *Whole Word Masking* durante el preentrenamiento. Alternativas competitivas incluyen XLM-RoBERTa [@conneau2020unsupervised] (multilingüe) y MarIA [@gutierrezfandino2022maria] (preentrenado sobre corpus de la Biblioteca Nacional de España).

Para adaptar BERT a ABSA, Sun et al. [@sun2019utilizing] propusieron la técnica de *auxiliary sentence*: la entrada al modelo se construye como `[CLS] reseña [SEP] aspecto [SEP]`, permitiendo que el token `[CLS]` codifique información específica del par (reseña, aspecto). Esta formulación, ampliada por Xu et al. [@xu2019bert] con preentrenamiento post-hoc sobre reseñas, alcanzó el estado del arte en SemEval-2014. Yang et al. [@yang2021multitask] extendieron el enfoque a multi-tarea conjunta ATE+ASC.

## 3.5 Modelos de Lenguaje a Gran Escala y prompting

Los modelos de lenguaje a gran escala, como GPT-3 [@brown2020language], LLaMA [@touvron2023llama] y Claude, alteraron el paradigma supervisado: en lugar de afinar pesos sobre datos etiquetados, se proporciona la tarea como texto y, opcionalmente, algunos ejemplos en contexto (*few-shot prompting*). Wei et al. [@wei2022chain] mostraron que cadenas de razonamiento explícitas pueden mejorar el desempeño en tareas complejas; en este proyecto, sin embargo, se solicita salida JSON directa para reducir variabilidad y facilitar evaluación automática.

Zhang et al. [@zhang2024sentiment] proveen una evaluación crítica del desempeño de los LLM en análisis de sentimientos: zero-shot tiende a ser competitivo en tareas simples (polaridad documento-nivel) pero rezagado en ABSA cuando la granularidad es alta. Este matiz motiva la inclusión de los LLM como uno de los tres brazos comparativos, no como reemplazo automático del fine-tuning.

## 3.6 Métricas de evaluación

Para la subtarea de clasificación de polaridad por aspecto utilizamos **F1-macro**, **F1-micro**, **precisión** y **recall** [@powers2011evaluation; @sokolova2009systematic]. F1-macro es prioritaria dado el desbalance esperado (clase positiva sobre-representada en reseñas de productos vendidos). Para las puntuaciones de reputación agregadas utilizamos **MAE** (Error Absoluto Medio), **RMSE** y la **correlación de Pearson** contra la referencia derivada del promedio de polaridades por aspecto en un subconjunto anotado manualmente.

# 4. Aparato crítico (los otros enfoques)

Reconstruimos críticamente la literatura previa, exhibiendo en qué condiciones cada familia de métodos resulta insuficiente para reputación basada en aspectos en español.

## 4.1 Enfoques basados puramente en lexicones

Trabajos pioneros como el de Hu y Liu [@hu2004mining] establecieron la viabilidad de extraer aspectos vía frecuencia de sustantivos y propagar polaridad léxica. Pérez-Rosas et al. [@perezrosas2012learning] extendieron el repertorio léxico a español. Sin embargo, estos enfoques presentan tres debilidades sistémicas:

1. **Insensibilidad al contexto sintáctico:** la negación (*"no me gustó nada"*) y la modulación por adverbios degradadores (*"poco confiable"*) requieren reglas explícitas que no escalan.
2. **Falta de cobertura léxica:** términos coloquiales mexicanos (*chido*, *chafa*, *a toda madre*) no aparecen en lexicones genéricos.
3. **Ambigüedad cruzada por aspecto:** la misma palabra puede tener polaridad opuesta según el aspecto. *Tarde* es negativo para *envío* pero neutro para *calidad* [@schouten2016survey].

## 4.2 ABSA con modelos clásicos de aprendizaje supervisado

Al añadir clasificadores SVM, Random Forest o LightGBM sobre representaciones TF-IDF o *embeddings* word2vec, se gana sensibilidad al contexto local pero se pierde generalización a expresiones no vistas. Además, la dependencia de ingeniería de características (lemas, POS tags, *n-grams*) introduce un costo metodológico no trivial [@bird2009nltk; @pedregosa2011scikit]. El umbral de desempeño típico en SemEval-2014 con SVM se encuentra entre 72% y 78% F1 [@pontiki2014semeval], notablemente por debajo de los transformers.

## 4.3 ABSA con redes neuronales pre-transformer

Antes de BERT, modelos basados en LSTM con mecanismos de atención (ATAE-LSTM, IAN, RAM) alcanzaron resultados competitivos en SemEval-2016 [@pontiki2016semeval; @pontiki2015semeval]. Limitaciones: requieren *embeddings* preentrenados independientes (GloVe, word2vec), no capturan dependencias bidireccionales como BERT [@devlin2019bert], y son sensibles al tamaño del corpus de entrenamiento ABSA específico (típicamente menos de 5 000 muestras).

## 4.4 BERT y derivados sin adaptación específica al dominio

Aplicar BERT *out-of-the-box* (sin *fine-tuning*, solo extracción de características) tiende a no alcanzar el potencial del modelo. El trabajo de Sun et al. [@sun2019utilizing] mostró que la formulación correcta de la entrada (par reseña-aspecto) marca una diferencia de hasta 5 puntos F1. Xu et al. [@xu2019bert] demuestran que un paso de post-entrenamiento sobre datos del dominio (reseñas) añade otros 3 a 4 puntos. Yang et al. [@yang2021multitask] integran ATE y ASC en una sola red multi-tarea.

## 4.5 LLM con prompting zero-shot

Aplicar GPT-4, Claude o LLaMA sin ejemplos en contexto sobre ABSA en español es atractivo (cero etiquetas) pero subóptimo: Zhang et al. [@zhang2024sentiment] reportan caídas de 5 a 10 puntos F1 respecto a few-shot bien diseñado, especialmente cuando el dominio tiene jerga local. Adicionalmente, el costo de inferencia por reseña puede ser dos órdenes de magnitud mayor que un BETO afinado en GPU.

## 4.6 Posicionamiento del presente trabajo

A diferencia de los anteriores, nuestro trabajo: 1) opera sobre español mexicano (no solo español peninsular dominante en la literatura europea); 2) integra el flujo desde texto crudo hasta puntuación reputacional, no únicamente la clasificación; 3) compara directamente las tres familias bajo el mismo split y métrica; y 4) reporta costos económicos y energéticos, dimensión poco común en la literatura [@zhang2024sentiment].

# 5. Setup Experimental

## 5.1 Hardware

Los experimentos se ejecutaron en tres entornos:

- **Entorno A (entrenamiento BETO):** GPU NVIDIA Tesla T4 (16 GB VRAM) proporcionada por Google Colab Pro, 25 GB de RAM, CUDA 12.1.
- **Entorno B (lexicones, SVM, evaluación, agregación):** CPU Intel i7-12700H, 16 GB de RAM, en una laptop personal.
- **Entorno C (LLM):** llamadas vía API directa de Anthropic al modelo Claude Sonnet 4.5 (`claude-sonnet-4-5`) desde la laptop personal.

## 5.2 Software

Toda la pipeline se implementó en Python 3.11. Las dependencias principales son `transformers==4.38`, `torch==2.1.0`, `scikit-learn==1.4` [@pedregosa2011scikit], `spaCy==3.7` con modelo `es_core_news_lg`, `nltk==3.8` [@bird2009nltk], y `anthropic==0.25` para el cliente del LLM. La lista completa de dependencias y sus versiones se versiona en `requirements.txt` y `pyproject.toml`. El código completo, junto con los notebooks reproducibles, las pruebas automatizadas y los scripts de entrenamiento, se encuentra en el repositorio:

> \url{https://github.com/chochy2001/aspect-based-reputation-analysis}

## 5.3 Objetivo y forma de evaluar

El objetivo es comparar empíricamente los tres enfoques (clásico, BETO, LLM) en dos tareas concatenadas:

1. **Clasificación de polaridad por aspecto (ASC):** granularidad reseña × aspecto. Métrica primaria F1-macro; secundarias accuracy, precisión, recall y matriz de confusión [@powers2011evaluation; @sokolova2009systematic].
2. **Construcción de reputación por aspecto:** agregación a puntuación 0 a 5 por producto y aspecto. Métricas MAE, RMSE y correlación de Pearson contra la referencia derivada del promedio de polaridades anotadas manualmente.

Se construyó un corpus piloto de **1 500 reseñas en español** etiquetadas manualmente por los tres autores con tripletas `(aspecto, polaridad)` cuando aplicaba, partiendo de una muestra mayor de Amazon México y MercadoLibre. El acuerdo inter-anotador medido con $\kappa$ de Cohen fue de 0.78, considerado sustancial en la escala de Landis y Koch. Se definieron cinco aspectos cerrados: **calidad**, **precio**, **envío**, **durabilidad** y **atención**. El split se realizó por producto (no por reseña) para evitar fuga de información: 70 % entrenamiento, 15 % validación y 15 % prueba.

# 6. Experimentos

## 6.1 Experimento 1: enfoque clásico (lexicón + SVM)

**Material.** Subconjunto de 1 050 reseñas (70 % del corpus anotado) con etiquetas a nivel `(reseña, aspecto, polaridad)`. Lexicón base: aproximadamente 3 200 entradas inspiradas en Pérez-Rosas et al. [@perezrosas2012learning], extendido con 280 términos coloquiales mexicanos curados manualmente.

**Método.** Para cada par (reseña, aspecto), se extrae una ventana de ±5 tokens alrededor de la mención del aspecto y se calcula un score léxico sumando los valores de polaridad ponderados por intensidad, con detección de negación dentro de los tres tokens previos. Posteriormente, se entrena un SVM lineal (LinearSVC) con representación TF-IDF (1-gram y 2-gram, *min_df*=3, *max_features*=10 000) sobre el texto completo concatenado con el aspecto, usando la implementación de scikit-learn [@pedregosa2011scikit].

**Evaluación y resultado.** Sobre el conjunto de prueba (225 reseñas, 1 008 menciones aspecto-reseña), el modelo alcanzó **F1-macro = 0.673**, accuracy = 0.732, precisión macro = 0.692, recall macro = 0.656. Los aspectos *envío* y *atención* obtuvieron F1 más bajos (0.61 y 0.64) por baja frecuencia de menciones (Tabla 1).

| Aspecto | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| Calidad | 0.74 | 0.71 | 0.725 | 412 |
| Precio | 0.69 | 0.68 | 0.685 | 198 |
| Envío | 0.65 | 0.58 | 0.613 | 142 |
| Durabilidad | 0.71 | 0.69 | 0.700 | 167 |
| Atención | 0.67 | 0.62 | 0.644 | 89 |
| **Macro** | **0.692** | **0.656** | **0.673** | **1 008** |

Tabla 1: Desempeño del enfoque clásico por aspecto. *Soporte* indica el número de menciones aspecto-reseña en el conjunto de prueba; las macros son el promedio aritmético simple por aspecto.

## 6.2 Experimento 2: BETO afinado con auxiliary sentence

**Material.** Mismo split que el Experimento 1. Modelo `dccuchile/bert-base-spanish-wwm-uncased` [@canete2020spanish], aproximadamente 110 M parámetros.

**Método.** Cada instancia se formula siguiendo Sun et al. [@sun2019utilizing]:

```
[CLS] La batería dura poco pero la pantalla es brillante [SEP] batería [SEP]
→ etiqueta: negativo
```

Se entrena durante tres épocas con AdamW (*lr* = 2e-5, *weight_decay* = 0.01), *batch_size* = 16 y *max_seq_length* = 128. Se usa truncación únicamente sobre el primer segmento para preservar el aspecto. Tiempo total de entrenamiento: 37 minutos en Tesla T4.

**Evaluación y resultado.** En el conjunto de prueba, BETO obtuvo **F1-macro = 0.846**, accuracy = 0.881, mostrando una mejora absoluta de 17.3 puntos F1-macro sobre el baseline clásico (Tabla 2).

| Aspecto | Precisión | Recall | F1 |
|---|---|---|---|
| Calidad | 0.89 | 0.88 | 0.885 |
| Precio | 0.85 | 0.83 | 0.840 |
| Envío | 0.82 | 0.81 | 0.815 |
| Durabilidad | 0.86 | 0.85 | 0.855 |
| Atención | 0.84 | 0.83 | 0.835 |
| **Macro** | **0.852** | **0.840** | **0.846** |

Tabla 2: Desempeño de BETO afinado por aspecto. Macros como promedio aritmético simple.

## 6.3 Experimento 3: LLM con few-shot prompting

**Material.** Mismo conjunto de prueba (225 reseñas, 1 008 menciones). API de Anthropic Claude Sonnet 4.5. Costo total: aproximadamente USD 4.20.

**Método.** Se construye un prompt con cuatro ejemplos en contexto (cubriendo casos positivo, mixto, multi-aspecto y neutro), siguiendo la formulación de Brown et al. [@brown2020language]. La instrucción solicita la polaridad en formato JSON estructurado para cada aspecto mencionado. Se usa temperatura 0 para reducir variabilidad.

**Evaluación y resultado.** Claude alcanzó **F1-macro = 0.825**, ligeramente inferior a BETO (-2.1 puntos) pero sin requerir entrenamiento. Llama la atención que el LLM superó a BETO en el aspecto *atención* (F1 = 0.860 frente a 0.835), probablemente por su capacidad de captar matices pragmáticos como *"el vendedor ni siquiera me contestó"* (Tabla 3) [@zhang2024sentiment].

| Aspecto | Precisión | Recall | F1 |
|---|---|---|---|
| Calidad | 0.85 | 0.86 | 0.855 |
| Precio | 0.83 | 0.81 | 0.820 |
| Envío | 0.80 | 0.78 | 0.790 |
| Durabilidad | 0.81 | 0.79 | 0.800 |
| Atención | 0.87 | 0.85 | 0.860 |
| **Macro** | **0.832** | **0.818** | **0.825** |

Tabla 3: Desempeño del LLM *few-shot* por aspecto.

## 6.4 Experimento 4: agregación a puntuaciones de reputación

**Material.** Las 225 reseñas del split de prueba, agrupadas por 47 productos únicos.

**Método.** Para cada producto $p$ y aspecto $a$, se agregan las predicciones de cada modelo mediante una media ponderada por confianza, suavizada hacia un prior neutro con factor $N/(N+k)$ donde $k=5$. Las confianzas son: probabilidad softmax para BETO, distancia normalizada al hiperplano para SVM, y consistencia heurística entre llamadas para el LLM. La fórmula completa se define en la Sección 3.2.

**Evaluación y resultado.** Se compara contra la puntuación de referencia derivada del promedio de polaridades anotadas manualmente (Tabla 4).

| Método | MAE | RMSE | Pearson |
|---|---|---|---|
| Lexicón + SVM | 0.832 | 1.024 | 0.612 |
| BETO | 0.413 | 0.581 | 0.847 |
| LLM | 0.476 | 0.638 | 0.812 |

Tabla 4: Agregación a puntuaciones de reputación (escala 0 a 5).

# 7. Resultado General

**Respuesta a H₀:** BETO con agregación ponderada produce reputaciones con MAE = 0.413, por debajo del umbral 0.5 establecido. **La hipótesis principal se valida.**

**Respuesta a H₁:** BETO supera al baseline clásico en 17.3 puntos de F1-macro (0.846 frente a 0.673), excediendo holgadamente el umbral de 10 puntos. **H₁ se valida.**

**Respuesta a H₂:** Claude Sonnet 4.5 con *few-shot* obtuvo F1-macro = 0.825, a 2.1 puntos de BETO, dentro del margen de 3 puntos hipotetizado. **H₂ se valida.**

En conjunto, las tres hipótesis se sostienen y la línea argumental del trabajo (BETO ofrece el mejor balance calidad-costo en español para ABSA, los LLM compiten sin entrenamiento, los lexicones siguen siendo útiles por interpretabilidad) queda respaldada por el piloto.

# 8. Análisis de Resultados

## 8.1 Por qué BETO domina al baseline clásico

La superioridad de BETO sobre el enfoque clásico (17.3 puntos F1) confirma resultados de la literatura general sobre transformers para ABSA [@sun2019utilizing; @xu2019bert; @zhang2022survey], pero adquiere matices interesantes en español mexicano. Análisis cualitativo de los errores del modelo clásico revela tres patrones predominantes:

1. **Negaciones de polaridad cruzada:** *"el producto no estaría tan mal si la atención fuera buena"*. El SVM acierta en *calidad* pero arrastra la polaridad negativa global a *atención*.
2. **Coloquialismos mexicanos:** expresiones como *"a toda madre la entrega"* (positivo sobre *envío*) no aparecen en lexicones convencionales y el SVM, sin las características adecuadas, falla [@perezrosas2012learning].
3. **Ironía y sarcasmo:** *"qué excelente servicio, llegó tres semanas tarde"*. Engaña al modelo léxico que ve "excelente" como positivo [@hutto2014vader].

BETO maneja (1) y (2) gracias a sus embeddings contextuales [@canete2020spanish; @devlin2019bert], pero también falla en (3): el sarcasmo sigue siendo un problema abierto [@zhang2022survey].

## 8.2 BETO frente al LLM: por qué el LLM no gana

A pesar de operar con un modelo considerablemente más grande que BETO en número de parámetros (cifra no publicada oficialmente por el proveedor, pero estimada en al menos uno o dos órdenes de magnitud por encima de los 110 M de BETO), Claude no superó a BETO en F1-macro. Hipotetizamos tres causas:

- **Adaptación al dominio:** BETO se afina explícitamente sobre el dominio de reseñas mexicanas; el LLM solo recibe cuatro ejemplos en contexto [@brown2020language; @zhang2024sentiment].
- **Calibración de confianza:** los softmax de BETO se utilizan directamente como peso en la agregación; las "confianzas" del LLM son aproximaciones heurísticas.
- **Ruido en clases minoritarias:** el LLM tiende a producir falsos positivos en *atención*, posiblemente por sobre-asociar el término con interacción humana.

No obstante, el LLM mostró ventajas claras en costo de desarrollo (cero entrenamiento) y flexibilidad para adaptar la instrucción. Estas ventajas pueden inclinar la balanza en escenarios de prototipado rápido o cuando no se dispone de GPU.

## 8.3 Análisis de la agregación de reputación

El paso de polaridades a puntuaciones de reputación introduce un eje de error adicional: dos modelos pueden tener F1 similar pero MAE diverso si difieren en la distribución de confianzas. La diferencia BETO (MAE 0.413) frente a LLM (MAE 0.476) es desproporcionada respecto a la diferencia F1 (-2.1 puntos), lo que se explica por la mejor calibración del softmax de BETO frente a las heurísticas de confianza usadas con el LLM. Esto refuerza un punto teórico de la literatura de sistemas de reputación [@josang2007survey; @resnick2000reputation]: la cantidad y calidad de la evidencia, no solo la decisión binaria, gobierna la calidad de la agregación.

## 8.4 Costo económico, energético y de cómputo

Reportamos el costo de cada experimento, dimensión generalmente ignorada en la literatura ABSA (Tabla 5).

| Método | Tiempo entrenamiento | Tiempo inferencia (225 reseñas) | Costo USD |
|---|---|---|---|
| Lexicón + SVM | 9 min CPU | 0.4 s | 0.00 |
| BETO | 37 min GPU T4 | 11 s | 0.50 (Colab) |
| LLM | n/a | 8 min (latencia API) | 4.20 |

Tabla 5: Costo por método sobre el conjunto de prueba.

A escala (12 000 reseñas), el LLM costaría aproximadamente USD 224, mientras que BETO, una vez entrenado, las clasificaría en segundos por menos de 1 USD. El argumento económico favorece a BETO una vez superado el costo único de etiquetar datos.

## 8.5 Distribución por aspecto y desbalance de clases

El aspecto *atención* es el más raro (8.8 % de las menciones del conjunto de prueba) y, paradójicamente, aquel donde el LLM brilla. Esto sugiere que en escenarios de *low-resource per-class*, los modelos con conocimiento previo amplio mantienen desempeño mientras que los afinados sobre clases minoritarias sufren [@zhang2024sentiment]. Este hallazgo es relevante para escenarios reales donde los datos suelen ser muy desbalanceados [@sokolova2009systematic].

## 8.6 Limitaciones internas del piloto

- El corpus anotado de 1 500 reseñas es modesto frente a SemEval-2014 (más de 6 000 instancias); convendría triplicarlo en una iteración posterior [@pontiki2014semeval].
- Los cinco aspectos predefinidos podrían no capturar dimensiones idiosincráticas (por ejemplo, *olor* en productos de hogar) [@schouten2016survey].
- El corpus se sesga hacia español mexicano; productos con reseñas argentinas, españolas o colombianas podrían rendir distinto [@gutierrezfandino2022maria].
- Se reporta una sola semilla por modelo. Una versión publicable debería reportar media y desviación estándar sobre al menos cinco semillas, con prueba de significancia (McNemar) para diferencias entre modelos.

# 9. Discusión y Trabajo Futuro

## 9.1 Implicaciones prácticas

Para un equipo que desee implementar análisis de reputación basado en aspectos en español hoy, la recomendación se desprende del análisis anterior:

- **Si hay 1 000 o más reseñas anotables:** afinar BETO. Ofrece el mejor balance entre calidad, costo y latencia, con métricas calibradas para agregación.
- **Si no hay datos anotados ni presupuesto para etiquetar:** LLM con *few-shot* bien construido, vigilando costos de API y privacidad de datos [@brown2020language; @zhang2024sentiment].
- **Si el principal requisito es transparencia y auditabilidad:** lexicones más SVM, aceptando 17 puntos F1 menos pero ganando un sistema enteramente inspeccionable [@hutto2014vader; @pedregosa2011scikit].

## 9.2 Trabajo futuro

1. **ABSA generativo** con modelos *sequence-to-sequence* en español (mT5, BLOOM-es), siguiendo el paradigma unificado de Zhang et al. [@zhang2022survey].
2. **Detección automática de aspectos** mediante clustering no supervisado de menciones, eliminando la dependencia del esquema cerrado [@hu2004mining; @schouten2016survey].
3. **Robustez a sarcasmo e ironía**, incorporando datasets específicos de figuras retóricas en español [@perezrosas2012learning].
4. **Sistemas de reputación dinámicos** que ponderen reseñas según fecha, reputación del autor y verificación de compra, alineados con la literatura clásica [@resnick2000reputation; @josang2007survey].
5. **Evaluación human-in-the-loop** del impacto real sobre decisiones de compra mediante experimentos A/B controlados.
6. **Modelos de español multidominio** como MarIA [@gutierrezfandino2022maria] frente a BETO [@canete2020spanish], comparando rendimiento por variante regional.
7. **Transferencia cross-lingual** con XLM-RoBERTa [@conneau2020unsupervised] desde datasets ABSA en inglés (SemEval-2014/2015/2016) hacia español [@pontiki2014semeval; @pontiki2015semeval; @pontiki2016semeval].
8. **Reporte estadístico riguroso:** múltiples semillas, pruebas de significancia, intervalos de confianza por bootstrap.

## 9.3 Cierre

El presente trabajo, lejos de zanjar la pregunta sobre el "mejor" enfoque ABSA en español, expone que la elección depende de variables económicas, regulatorias y operativas tanto como técnicas. La hipótesis principal se valida en el piloto, pero la lección de fondo es metodológica: tres familias muy distintas producen resultados sorprendentemente cercanos, y el verdadero diferencial está en la línea de agregación, donde la calibración de confianza supera en importancia a la precisión bruta del clasificador subyacente [@josang2007survey].

# Bibliografía

::: {#refs}
:::
