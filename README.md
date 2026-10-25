# Granulometría de suelos a partir de fotos

> Predigo la curva granulométrica acumulada de una muestra de suelo a partir de una foto de su superficie.
> Competición de Kaggle lanzada por el proyecto de investigación GRID.

<!-- GIF de la demo aquí -->

## Resultados

Métrica oficial: EMD con peso logarítmico (área entre curvas en escala log, 0-500, menor es mejor). Validación *leave-one-sample-out* sobre las 24 muestras de entrenamiento.

| Modelo | EMD (LOSO) | EMD cambiando de cámara* | Leaderboard |
|---|---|---|---|
| Uniforme (baseline de Kaggle) | 102,0 | – | – |
| Curva media del train | 89,9 | – | – |
| Ridge, todas las features de textura | **41,5** | 83,3 / 71,9 | – |
| Ridge, sin color ni brillo (envío) | 44,2 | **70,2 / 53,2** | 64,9 (público, ~367 de 658) |
| CNN (EfficientNet, salida monótona) | – | – | – |

\* Entrenando con Motorola y evaluando con Samsung / al revés. El test es todo iPhone, así que este es el escenario realista: la puntuación pública (64,9) cae justo en ese rango, mientras que la validación con la misma cámara (44) era optimista.

**Demo:** _enlace a Hugging Face Spaces_

## Hallazgos clave

- **Las fotos de entrenamiento venían reducidas** (de 2,5 a 6 veces). Sin corregir los píxeles por mm, la escala física sale mal. Tras reescalar a 4,5 px/mm, todas las fotos abarcan entre 290 y 350 mm.
- **La cámara es el mayor riesgo.** La misma muestra fotografiada con dos móviles da distinto color y contraste, y un modelo que usa color empeora mucho al cambiar de cámara.
- **La foto no siempre coincide con el laboratorio:** hay terrones de finos que parecen grava y gravas enterradas en la matriz.

<!-- Gráficos: reports/figures/04_ejemplos_por_clase.png, 07_baseline_emd.png, 09_curvas_predichas.png -->

## Enfoque

1. **EDA**: indicadores D10/D30/D50/D60, Cu y Cc, clasificación en grava, arena o finos, y revisión de las imágenes.
2. **Baseline**: features de textura (GLCM, bordes, histogramas) con LightGBM.
3. **CNN**: backbone preentrenado que predice incrementos con softmax y los acumula, así la curva siempre es monótona.
4. **Análisis de errores**: dónde falla el modelo según el tipo de suelo.

## Cómo reproducirlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/download_data.sh      # necesita ~/.kaggle/kaggle.json
```

Los datos no están en el repo porque las reglas de la competición no permiten redistribuirlos.

## Limitaciones y próximos pasos

-
