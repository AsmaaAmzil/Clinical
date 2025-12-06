import importlib.util
from pathlib import Path


# Tests basiques pour le module core/optimizer.py
# À adapter selon les fonctions/classes réelles dans optimizer.py

def test_import_optimizer_module():
    """Vérifie que le module optimizer peut être chargé sans erreur."""
    project_root = Path(__file__).resolve().parents[1]
    optimizer_path = project_root / "core" / "optimizer.py"

    # Le fichier doit exister
    assert optimizer_path.exists(), f"Fichier introuvable: {optimizer_path}"

    # Charger le module depuis son chemin de fichier
    spec = importlib.util.spec_from_file_location("optimizer", optimizer_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


def test_dummy_example():
    """Exemple de test très simple pour vérifier que pytest fonctionne."""
    assert 1 + 1 == 2
