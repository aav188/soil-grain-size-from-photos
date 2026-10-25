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
| CNN congelada (ConvNeXt-T) + cabeza monótona entrenada con EMD | 46,7 | **42,7 / 42,0** | 72,3 (público) |

\* Entrenando con Motorola y evaluando con Samsung / al revés. El test es todo iPhone, así que este es el escenario realista: la puntuación pública (64,9) cae justo en ese rango, mientras que la validación con la misma cámara (44) era optimista.

**La CNN gana en validación y pierde en Kaggle.** Mi hipótesis es que la validación entre cámaras no reproduce todo el cambio de dominio. Las fotos de train (Motorola y Samsung) llegaron ya reducidas y recomprimidas en JPEG, mientras que las de test (iPhone) vienen a resolución completa y las reduzco yo, así que la textura fina que ve la CNN es distinta. La CNN, además, hace predicciones tajantes (7 de 10 muestras como suelo casi todo fino), mientras que el baseline reparte más el riesgo. Con 10 muestras de test la puntuación pública es ruidosa, pero la diferencia es coherente con esa explicación. Siguiente paso: imitar en las fotos de test la cadena de procesado del train (reducción y recompresión JPEG) y evaluar con una validación que la incluya.

**Demo:** app en Streamlit (`app/app.py`). Subes una foto, eliges el móvil y devuelve la curva, D10–D60, Cu, Cc y la clase. Ver *Cómo reproducirlo*.

## Hallazgos clave

- **Las fotos de entrenamiento venían reducidas** (de 2,5 a 6 veces). Sin corregir los píxeles por mm, la escala física sale mal. Tras reescalar a 4,5 px/mm, todas las fotos abarcan entre 290 y 350 mm.
- **La cámara es el mayor riesgo.** La misma muestra fotografiada con dos móviles da distinto color y contraste, y un modelo que usa color empeora mucho al cambiar de cámara.
- **La CNN aguanta el cambio de cámara mucho mejor que la textura hecha a mano** (EMD entre cámaras de 61,7 a 42,3), y entrenar con la métrica oficial importa más que el modelo que va encima.
- **La foto no siempre coincide con el laboratorio:** hay terrones de finos que parecen grava y gravas enterradas en la matriz.

<!-- Gráficos: reports/figures/04_ejemplos_por_clase.png, 07_baseline_emd.png, 09_curvas_predichas.png -->

## Enfoque

1. **EDA**: indicadores D10/D30/D50/D60, Cu y Cc, clasificación en grava, arena o finos, y revisión de las imágenes.
2. **Baseline**: features de textura (GLCM, energía por escala, gradientes) a escala física común, con Ridge y LightGBM. Se descartan el color y el brillo absolutos porque dependen de la cámara.
3. **CNN**: ConvNeXt-Tiny preentrenada y congelada sobre recortes de 50×50 mm, con una cabeza que predice incrementos con softmax y los acumula (la curva siempre es monótona), entrenada directamente con la EMD como pérdida.
4. **Análisis de errores**: dónde falla el modelo según el tipo de suelo.

## Cómo reproducirlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/download_data.sh      # necesita ~/.kaggle/access_token o ~/.kaggle/kaggle.json
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb   # EDA, baseline y CNN (GPU recomendada)
(cd src && python -m soilgrain.train_final)                             # modelo final para la demo
streamlit run app/app.py
```

Los datos no están en el repo porque las reglas de la competición no permiten redistribuirlos.

## Limitaciones y próximos pasos

-
