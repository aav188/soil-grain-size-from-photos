#!/usr/bin/env bash
# Descarga los datos de la competición con la API de Kaggle.
# Requisitos: pip install kaggle y ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token).
# Antes hay que aceptar las reglas de la competición en la web.
set -euo pipefail

COMPETITION="${COMPETITION:-soil-grain-size-from-photos}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"

mkdir -p "$DEST"
kaggle competitions download -c "$COMPETITION" -p "$DEST"
unzip -o -q "$DEST/$COMPETITION.zip" -d "$DEST"
rm "$DEST/$COMPETITION.zip"
echo "Datos en $DEST"
