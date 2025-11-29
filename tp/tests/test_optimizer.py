import pytest
import sys
from pathlib import Path


# Tests basiques pour le module core/optimizer.py
# À adapter selon les fonctions/classes réelles dans optimizer.py

def test_import_optimizer_module():
    """Vérifie que le module optimizer peut être importé sans erreur."""
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import core.optimizer  # noqa: F401


def test_dummy_example():
    """Exemple de test très simple pour vérifier que pytest fonctionne."""
    assert 1 + 1 == 2
