# TP Final — Detección de sub-declaración de ingresos (ARCA)

**Aprendizaje Automático, 2026 Q1.**

## Contexto

Somos el equipo de análisis de riesgo fiscal de ARCA. El objetivo es entrenar un modelo que estime la probabilidad de que un declarante que reportó ingresos menores a USD 50.000 anuales en realidad gane más que ese umbral.

**El modelo no determina infracción.** Rankea a los declarantes por riesgo estimado, para que el equipo de auditoría priorice mejor a quién revisar primero. El sistema actual es 100% manual (detección y auditoría humana); el modelo lo complementa, no lo reemplaza.

Dataset: [*Adult / Census Income*](https://archive.ics.uci.edu/dataset/2/adult) (UCI Machine Learning Repository), reinterpretado para este escenario ficticio. 32.561 declarantes, 15 columnas originales.

## Instalación y reproducción

El notebook arma su propio entorno (primeras dos celdas): crea un `.venv`, instala las dependencias (`requirements.txt` se genera automáticamente) y registra un kernel de Jupyter (`Python (tp-arca)`). Pasos:

```bash
git clone <este repo>
cd <este repo>
jupyter lab tp_final_arca.ipynb
```

Dentro del notebook:
1. Correr las dos primeras celdas (sección "0. Entorno y dependencias").
2. Seleccionar el kernel **`Python (tp-arca)`** desde el selector de Jupyter.
3. `Restart & Run All`.

Dependencias: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `ipykernel`, `joblib`.

## Estructura del repo

```
.
├── README.md
├── adult.csv               # dataset original (sin header)
├── extra-data.csv          # lote externo etiquetado para simulación de producción
├── model_serving.py        # validación, persistencia e inferencia standalone
├── artifacts/
│   └── random_forest_income.joblib  # pipeline completo ya entrenado
└── tp_final_arca.ipynb     # notebook con todo el pipeline
```

## Estado actual

Completado (Parte 3 de la consigna):

1. Carga de datos y EDA (nulos, balance de clases, correlaciones numéricas y categóricas).
2. Limpieza y selección de variables (ver decisiones abajo).
3. Separación train/test estratificada.
4. Preprocesamiento (`ColumnTransformer`: escalado + one-hot) dentro de un `Pipeline` de sklearn.
5. Baseline y primeros modelos, comparados por cross-validation.
6. Predicciones con probabilidad y ranking de riesgo (ilustrado out-of-fold).
7. Métricas de negocio a medida del caso de uso.
8. Búsqueda de hiperparámetros (`RandomizedSearchCV`, scoring múltiple).

También están completados el análisis de errores y cohortes, la evaluación final
sobre test, la Parte 4, la inferencia standalone y la simulación de monitoreo y
reentrenamiento con datos externos. Quedan la Parte 5 y las presentaciones.

## Decisiones de diseño importantes

### Variable objetivo

`income` (`>50K` / `<=50K`) se usa como proxy de "sub-declaración": el modelo estima la probabilidad de que un declarante que reportó menos del umbral en realidad gane más.

### Variables descartadas

| Variable | Motivo |
|---|---|
| `fnlwgt` | Peso muestral del censo. No tiene sentido a nivel individuo y no correlaciona con el target (r≈-0.01, no significativo). |
| `race` | No se usa institucionalmente en Argentina para este tipo de análisis. Asociación débil con el target (V de Cramér ≈ 0.10). |
| `native_country` | Asociación débil (V≈0.10, similar a `race`); ~90% de los casos son `United-States` y el resto se reparte en 40 categorías chicas — casi no aporta variabilidad, y el desbalance geográfico responde al contexto de EE.UU., no al argentino. |
| `hours_per_week` | Sí es predictiva (r≈0.23), pero se descarta igual: no es un dato disponible en una declaración impositiva real. Se prioriza usar variables que existirían en producción por sobre una métrica más alta con una variable que no vamos a tener. |
| `education` | Redundante: mapeo 1 a 1 con `education_num` (versión ordinal, ya numérica). |

`sex` se mantiene como feature (tiene poder predictivo real, V≈0.22) pero queda marcada como **variable sensible a monitorear** — ver "Hallazgos" abajo.

### Valores faltantes

`workclass` y `occupation` usan `"?"` como marca de faltante (5.6% cada una), y faltan casi siempre juntas (1836 de 1843 casos). Como el patrón no es aleatorio, se imputan con una categoría explícita `"Sin_dato"` en vez de imputar por la moda o eliminar filas — así se preserva la señal de que el dato falta, en vez de esconderla.

### Separación train/test

**Estratificada** (80/20) por el target, para que el desbalance de clases (~76/24) quede igual de representado en train y en test.

### Preprocesamiento

`ColumnTransformer` dentro de un `Pipeline` de sklearn: `StandardScaler` para numéricas, `OneHotEncoder(handle_unknown="ignore")` para categóricas. El ajuste (`fit`) del transformer se hace solo con train, para no filtrar información de test.

### Principio metodológico: test se toca una única vez

Todo el desarrollo — comparación inicial de modelos, validación de las métricas de negocio, búsqueda de hiperparámetros — se hace con **cross-validation sobre train**. El test set se evalúa una sola vez, al final, para reportar los resultados definitivos. Esto evita que el test termine influyendo (aunque sea indirectamente) en las decisiones de diseño.

### Métricas de negocio a medida

Además de accuracy/precision/recall/F1/ROC-AUC, se definieron dos métricas pensadas específicamente para cómo se usaría el modelo (un ranking, no una clasificación binaria):

- **`weighted_precision`**: fracción de la "masa de probabilidad" repartida por el modelo que efectivamente cae sobre sub-declarantes reales — `Σ(pᵢ·yᵢ) / Σ(pᵢ)`. Simula un escenario donde el esfuerzo de auditoría se reparte en proporción a la probabilidad estimada, no con un corte binario.
- **`precision_at_n`**: de los N declarantes con mayor probabilidad estimada, qué fracción son sub-declarantes reales (calculada para N=100 y N=1000). Más simple de explicar a alguien no técnico.

### Modelos e hiperparámetros

Se probaron regresión logística y Random Forest. Con hiperparámetros por defecto (evaluados por CV), la regresión logística superaba al Random Forest — señal de que el RF sin tunear sobreajustaba sobre el espacio disperso de las variables one-hot. Se tunearon ambos con `RandomizedSearchCV` (scoring múltiple: ROC-AUC como criterio principal, más las métricas de negocio a modo diagnóstico), y ahí el Random Forest pasó a competir de igual a igual.

**Modelo elegido: Random Forest tuneado** (`n_estimators=200, max_depth=16, min_samples_leaf=5`), por su ventaja en la validación OOF realizada exclusivamente sobre train. La regresión logística queda como alternativa más simple e interpretable para la Parte 5.

---

## Actualización: validación OOF, análisis de errores y cohortes

> **Nota:** esta sección agrega el trabajo realizado posteriormente y debe leerse como una actualización del apartado **Estado actual**. No se modificó el contenido anterior del README.

### Aclaración metodológica

La elección del modelo **no utilizó el conjunto de test**. La comparación y selección entre la regresión logística y el Random Forest se realizó mediante validación cruzada sobre el conjunto de entrenamiento.

El conjunto de test fue utilizado una única vez para la evaluación final y no se
empleó para modificar el modelo, sus hiperparámetros ni el umbral.

### Trabajo agregado al notebook

Se incorporaron las siguientes etapas al pipeline:

1. Evaluación out-of-fold de los modelos tuneados.
2. Comparación mediante métricas tradicionales y métricas de ranking.
3. Matriz de confusión del Random Forest tuneado.
4. Análisis de falsos positivos y falsos negativos.
5. Análisis de los errores de mayor confianza.
6. Análisis de rendimiento por cohortes.
7. Análisis del comportamiento del top 1000 del ranking.
8. Interpretación del Random Forest mediante importancia de variables.

Las predicciones out-of-fold permiten analizar los errores sin utilizar test y sin evaluar cada observación con un modelo que haya sido entrenado con esa misma observación.

### Comparación de modelos tuneados

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Precision@1000 | Recall@1000 | Lift@1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Regresión logística | 0.832 | 0.722 | 0.581 | 0.644 | 0.889 | 0.760 | 0.961 | 0.205 | 3.683 |
| Random Forest | **0.848** | **0.796** | 0.559 | **0.657** | **0.904** | **0.804** | **0.997** | **0.213** | **3.821** |

El Random Forest tuneado presenta el mejor desempeño global y de ranking. La regresión logística alcanza un recall ligeramente mayor con umbral 0.5, pero el Random Forest logra mejor precision, F1, ROC-AUC, PR-AUC y rendimiento dentro del top 1000.

### Interpretación del ranking

La `Precision@1000` del Random Forest fue **0.997**. Esto significa que, entre los 1000 casos con mayor riesgo estimado, aproximadamente **997 pertenecen realmente a la clase positiva**.

El `Recall@1000` fue **0.213**, por lo que esos 1000 casos concentran aproximadamente el **21.3% de todos los positivos** disponibles en la validación out-of-fold.

El `Lift@1000` fue **3.821**. La precision del top 1000 es, por lo tanto, aproximadamente **3.8 veces mayor que la que se obtendría seleccionando casos al azar**.

Esto indica que el modelo es especialmente útil como herramienta de priorización cuando la cantidad de auditorías disponibles es limitada. No debe interpretarse como un sistema capaz de detectar la totalidad de los casos positivos.

### Matriz de confusión OOF del Random Forest

Con un umbral de 0.5 se obtuvieron:

- 12.588 verdaderos negativos;
- 2.619 verdaderos positivos;
- 672 falsos positivos;
- 2.062 falsos negativos.

El modelo obtiene una precision alta, pero deja sin detectar una proporción relevante de los positivos. Esto es consistente con su uso como ranking de priorización y no como decisión automática.

### Análisis de falsos positivos

Los falsos positivos se concentran principalmente en perfiles asociados por el modelo con la clase positiva: personas casadas, con educación alta y con ocupaciones profesionales o gerenciales.

Hallazgos principales:

- El **98.8%** de los falsos positivos pertenece a `Married-civ-spouse`.
- `Prof-specialty` representa el **37.1%** de los falsos positivos.
- `Exec-managerial` representa el **29.6%**.
- La tasa de falsos positivos fue **0.261** para personas con posgrado.
- La tasa de falsos positivos fue **0.154** para personas con grado universitario.
- Por ocupación, las mayores FPR aparecen en `Prof-specialty` (**0.176**), `Exec-managerial` (**0.158**) y `Tech-support` (**0.133**).
- La categoría laboral `Self-emp-inc` presenta una FPR de **0.215**.

Estos errores podrían producir auditorías innecesarias sobre personas con perfiles similares a los de la clase positiva. La predicción debe utilizarse únicamente para ordenar revisiones posteriores realizadas por personas.

### Análisis de falsos negativos

Los falsos negativos presentan menor educación y valores mucho menores de `capital_gain` que los verdaderos positivos.

- `education_num` promedio de falsos negativos: **10.59**.
- `education_num` promedio de verdaderos positivos: **12.46**.
- `capital_gain` promedio de falsos negativos: **226.21**.
- `capital_gain` promedio de verdaderos positivos: **8896.63**.

Las mayores tasas de falsos negativos por ocupación fueron:

| Ocupación | FNR |
|---|---:|
| `Handlers-cleaners` | 0.831 |
| `Other-service` | 0.829 |
| `Machine-op-inspct` | 0.745 |
| `Farming-fishing` | 0.713 |
| `Transport-moving` | 0.710 |

También se observó:

- FNR de **0.716** en personas de hasta 25 años.
- FNR de **0.788** en personas con secundario incompleto.
- FNR de **0.664** en casos sin datos laborales.

El modelo tiene dificultades para detectar positivos que no muestran los indicadores económicos, educativos o laborales más comunes de la clase positiva. En una aplicación real deberían mantenerse mecanismos alternativos de selección para no excluir sistemáticamente estos casos.

### Análisis por cohortes

#### Sexo

| Cohorte | Recall | FPR | FNR | Tasa de selección top 1000 | Prevalencia positiva |
|---|---:|---:|---:|---:|---:|
| Mujeres | 0.497 | 0.020 | 0.503 | 0.012 | 0.138 |
| Hombres | 0.574 | 0.072 | 0.426 | 0.080 | 0.328 |

El modelo deja sin detectar una proporción mayor de positivos entre las mujeres. Los hombres, por otro lado, presentan una tasa mayor de falsos positivos y son seleccionados con mucha mayor frecuencia dentro del top 1000.

Parte de esta diferencia se relaciona con la distinta prevalencia de la clase positiva en el dataset, aunque el comportamiento debería monitorearse en una implementación real.

#### Edad

La mayor tasa de falsos negativos aparece en personas de hasta 25 años (**0.716**), seguida por el grupo de 26 a 35 años (**0.543**).

Las bandas de 36 a 45 y de 46 a 55 años son seleccionadas con mayor frecuencia en el top 1000, con tasas de **8.5%** y **9.6%**, respectivamente.

#### Educación

La tasa de selección dentro del top 1000 aumenta fuertemente con el nivel educativo:

- Secundario incompleto: **0.2%**.
- Secundario completo: **2.5%**.
- Grado universitario: **11.2%**.
- Posgrado: **19.0%**.

Este comportamiento coincide parcialmente con las diferencias de prevalencia entre las cohortes, pero también muestra que el ranking depende considerablemente de la educación.

### Importancia de variables

Las variables con mayor importancia agregada en el Random Forest fueron:

| Variable | Importancia |
|---|---:|
| `capital_gain` | 0.239 |
| `marital_status` | 0.194 |
| `relationship` | 0.146 |
| `education_num` | 0.145 |
| `occupation` | 0.082 |
| `age` | 0.070 |
| `capital_loss` | 0.069 |

Entre las categorías individuales más utilizadas aparecen `Married-civ-spouse`, `Husband`, `Never-married`, `Wife`, `Prof-specialty` y `Exec-managerial`.

Estas importancias representan asociaciones predictivas dentro del dataset. No demuestran causalidad ni deberían utilizarse individualmente para justificar una auditoría.

### Estado del proyecto después de esta actualización

Completado adicionalmente:

- evaluación OOF de los modelos tuneados;
- comparación mediante ROC-AUC, PR-AUC, F1 y métricas del ranking;
- análisis de la matriz de confusión;
- análisis de falsos positivos y falsos negativos;
- análisis de errores de alta confianza;
- análisis por cohortes;
- análisis del top 1000;
- interpretación mediante importancia de variables;
- conclusiones técnicas de la Parte 3.

### Trabajo pendiente para continuar

1. Desarrollar la Parte 5 en lenguaje no técnico.
2. Preparar la presentación técnica de 10 minutos.
3. Preparar la presentación no técnica de 5 minutos.

## Inferencia sobre datos nuevos

El artefacto contiene el pipeline completo ya ajustado. Por lo tanto, no hay que
recrear manualmente el one-hot encoding ni el escalado:

```python
from model_serving import predict_one

row = [
    47, "Private", 120939, "Some-college", 10,
    "Married-civ-spouse", "Tech-support", "Husband",
    "White", "Male", 0, 0, 45, "United-States",
]

probability = predict_one(row)
print(probability)  # 0.535945 aproximadamente
```

También se puede usar `predict_rows()` con un diccionario o un `DataFrame`. El
modelo estima `income >50K`; el umbral de decisión 0.5 es configurable y es
distinto del umbral de ingresos que define el target.

## Simulación con datos externos

`extra-data.csv` se divide de forma estratificada en 50% adaptación, 25%
validation y 25% test. El desarrollo V2 combina todos los datos históricos con
adaptación y usa K-fold para comparar reentrenamiento base y ponderación de
cohortes mediante accuracy, precision, recall, F1, ROC-AUC y PR-AUC.

Validation compara el productivo y los candidatos sobre las mismas 4.070 filas.
El reentrenado base mejoró ROC-AUC solo 0.001 y redujo recall y F1, por lo que no
cumplió la regla de promoción y se mantuvo el productivo. Test evaluó únicamente
ese modelo ya seleccionado sobre 4.071 filas; sus resultados no participaron de
la decisión.

## Referencias

- Consigna del TP y dataset: ver carpeta del curso.
- Presentación de la Entrega 1 (contexto, justificación del dataset, validada por la cátedra).
- Fuente del dataset: [UCI Machine Learning Repository — Adult](https://archive.ics.uci.edu/dataset/2/adult).
