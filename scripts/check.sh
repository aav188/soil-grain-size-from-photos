#!/usr/bin/env bash
# Comprobación rápida del proyecto: el equivalente al `make` sin warnings.
# No ejecuta los notebooks enteros (tardan minutos y necesitan los datos); comprueba que todo importa,
# que la métrica y las curvas se comportan como deben y que la app arranca.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

echo "1/4 Compilando src/ y app/"
"$PY" -W error -m compileall -q src app

echo "2/4 Comprobando métrica, curvas e indicadores"
"$PY" - <<'EOF'
import sys
sys.path.insert(0, "src")
import numpy as np
from soilgrain import data, embeddings, features, gradation, metrics, model, plotting, predict  # noqa: F401
from soilgrain.gradation import gradation_indicators, soil_class
from soilgrain.metrics import SIEVES, emd, make_valid

rng = np.random.default_rng(0)
curve = np.sort(rng.uniform(0, 100, (5, len(SIEVES))), axis=1)
curve[:, -1] = 100
assert np.allclose(emd(curve, curve), 0), "EMD de una curva consigo misma debe ser 0"
assert np.isclose(emd(np.zeros(11), np.full(11, 100.0))[0], 500), "EMD máxima debe ser 500"
valid = make_valid(rng.uniform(-20, 120, (5, len(SIEVES))))
assert (np.diff(valid, axis=1) >= 0).all() and valid.min() >= 0 and (valid[:, -1] == 100).all()
ind = gradation_indicators(SIEVES, [0, 2, 5, 12, 30, 55, 75, 90, 98, 100, 100])
assert 0.063 < ind["D50"] < 2 and ind["Cu"] > 1, ind
assert soil_class(SIEVES, [0, 2, 5, 12, 30, 55, 75, 90, 98, 100, 100]) == "arena"

import torch
from soilgrain.model import MonotoneHead
out = MonotoneHead(8)(torch.randn(4, 8))
assert (out.diff(dim=1) >= 0).all() and torch.allclose(out[:, -1], torch.full((4,), 100.0))
print("   ok")
EOF

echo "3/4 Comprobando que las celdas de los notebooks son Python válido"
"$PY" - <<'EOF'
import ast, glob, nbformat
for path in sorted(glob.glob("notebooks/*.ipynb")):
    for cell in nbformat.read(path, 4).cells:
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=path)
print("   ok")
EOF

echo "4/4 Arrancando la app sin foto"
# La salida se captura para poder filtrar avisos sin perder el código de error
if ! out=$(APP_FILE="${APP_FILE:-app/app.py}" "$PY" -c '
import os
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(os.environ["APP_FILE"], default_timeout=120).run()
assert not at.exception, [e.value for e in at.exception]
' 2>&1); then
    echo "$out" | grep -v -E "ScriptRunContext|HF_TOKEN"
    exit 1
fi
echo "   ok"
echo "Todo correcto"
