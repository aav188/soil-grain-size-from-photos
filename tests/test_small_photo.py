"""Una foto demasiado pequeña para un solo recorte debe dar un error claro, no uno de numpy."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soilgrain.features import photo_features


def test_photo_features_too_small():
    with pytest.raises(ValueError, match="menor que un recorte"):
        photo_features(np.zeros((50, 50, 3), dtype=np.uint8))


def test_photo_features_one_patch():
    img = np.random.default_rng(0).integers(0, 255, (180, 180, 3), dtype=np.uint8)
    assert photo_features(img)["n_patches"] == 1


def test_patch_embeddings_too_small_fails_before_model():
    pytest.importorskip("timm")
    from soilgrain.embeddings import patch_embeddings
    with pytest.raises(ValueError, match="hace falta al menos"):
        patch_embeddings(np.zeros((100, 300, 3), dtype=np.uint8), backbone=None)
