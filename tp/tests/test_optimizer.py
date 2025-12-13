import importlib.util
from pathlib import Path

import pytest


# Tests basiques pour le module core/optimizer.py
# À adapter selon les fonctions/classes réelles dans optimizer.py

def test_import_optimizer_module():
    """Vérifie que le module optimizer peut être chargé sans erreur."""
    # Cherche le fichier core/optimizer.py en remontant jusqu'à 5 niveaux
    current = Path(__file__).resolve()
    optimizer_path = None
    for _ in range(5):
        candidate = current / "core" / "optimizer.py"
        if candidate.exists():
            optimizer_path = candidate
            break
        current = current.parent

    if optimizer_path is None:
        pytest.skip("core/optimizer.py introuvable dans l'env de test")

    # Charger le module depuis son chemin de fichier
    spec = importlib.util.spec_from_file_location("optimizer", optimizer_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


def test_dummy_example():
    """Exemple de test très simple pour vérifier que pytest fonctionne."""
    assert 1 + 1 == 2
