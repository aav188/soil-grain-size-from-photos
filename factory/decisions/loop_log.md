# Diario del bucle de mejora del código

Una línea por iteración: qué se mejoró o por qué no se hizo nada. Sirve para no repetir trabajo.

- 2026-11-03 23:40 · Foto demasiado pequeña para un recorte (224 px CNN / 40 mm textura): error claro en `embeddings.patch_embeddings` y `features.photo_features` en vez del `need at least one array to stack` de numpy; la app lo muestra con `st.error`. Tests en `tests/test_small_photo.py`.
- 2026-11-03 23:55 · Experimento (con datos): textura centrada por familia de cámara (`validation.center_by_camera`, perfil `v3`). Proxies mejores (49,2 / 48,5 / 43,2), pero Kaggle 58,56 frente a 56,25 de v2. Resultado negativo: no centrar. v2 sigue siendo el envío de referencia.
- 2026-11-04 00:47 · Diagnóstico (con datos): Ridge Agregado solo en Kaggle = 43,52 (v2 56,25). Los miembros con CNN sesgan hacia finos. Pasa a ser el perfil por defecto (`agg`). Descartado también: PCA de curvas como salida (cambios en el nivel de ruido). El error se concentra en 0,063–2 mm, por debajo de la resolución de las fotos de train (0,22 mm/px).
