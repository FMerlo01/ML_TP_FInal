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

Dependencias: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `ipykernel`.

## Estructura del repo

```
.
├── README.md
├── adult.csv               # dataset original (sin header)
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

Pendiente: Análisis de errores y cohortes en validation, test y métricas. Luego partes 4 y 5.

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

**Modelo elegido: Random Forest tuneado** (`n_estimators=200, max_depth=16, min_samples_leaf=5`), por una ligera ventaja sobre la regresión logística tuneada en el resultado final de test. La regresión logística queda como alternativa más simple e interpretable para la Parte 5.

## Referencias

- Consigna del TP y dataset: ver carpeta del curso.
- Presentación de la Entrega 1 (contexto, justificación del dataset, validada por la cátedra).
- Fuente del dataset: [UCI Machine Learning Repository — Adult](https://archive.ics.uci.edu/dataset/2/adult).
