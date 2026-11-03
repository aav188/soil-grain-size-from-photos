# Diario del bucle de mejora del código

Una línea por iteración: qué se mejoró o por qué no se hizo nada. Sirve para no repetir trabajo.

- 2026-11-03 23:40 · Foto demasiado pequeña para un recorte (224 px CNN / 40 mm textura): error claro en `embeddings.patch_embeddings` y `features.photo_features` en vez del `need at least one array to stack` de numpy; la app lo muestra con `st.error`. Tests en `tests/test_small_photo.py`.
