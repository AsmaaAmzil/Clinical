import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.interface_clinique import ClinicalPredictor
from core.model import Model


class DummyProbModel(Model):
    """Dummy model that implements predict_proba only."""

    def __init__(self, proba: float):
        super().__init__()
        self.is_trained = True
        self._proba = float(proba)

    def predict_proba(self, X):  # override: ignore self.model
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        proba_pos = self._proba
        proba_neg = 1.0 - proba_pos
        return np.tile([proba_neg, proba_pos], (X.shape[0], 1))


class DummyLabelModel(Model):
    """Dummy model that only exposes predict (no predict_proba)."""

    def __init__(self, label: int):
        super().__init__()
        self.is_trained = True
        self._label = int(label)

    def predict(self, X):  # override: ignore self.model
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.full(shape=(X.shape[0],), fill_value=self._label, dtype=int)


def test_diagnose_uses_predict_proba_threshold():
    """If proba >= 0.5 -> 'Infecté', else 'Sain'."""
    patient = [1.0, 2.0, 3.0]

    inf_model = DummyProbModel(0.8)
    predictor_inf = ClinicalPredictor(inf_model)
    assert predictor_inf.diagnose(patient) == "Infecté"

    healthy_model = DummyProbModel(0.2)
    predictor_healthy = ClinicalPredictor(healthy_model)
    assert predictor_healthy.diagnose(patient) == "Sain"


def test_diagnose_falls_back_to_predict():
    """If predict_proba is not available, uses predict (0 -> Sain, 1 -> Infecté)."""
    patient = np.array([1.0, 2.0, 3.0])

    inf_model = DummyLabelModel(1)
    predictor_inf = ClinicalPredictor(inf_model)
    assert predictor_inf.diagnose(patient) == "Infecté"

    healthy_model = DummyLabelModel(0)
    predictor_healthy = ClinicalPredictor(healthy_model)
    assert predictor_healthy.diagnose(patient) == "Sain"


def test_diagnose_accepts_1d_input():
    """diagnose must accept 1D inputs (n_features,) and reshape internally."""
    model = DummyProbModel(0.7)
    predictor = ClinicalPredictor(model)

    result = predictor.diagnose([0.1, 0.2, 0.3])
    assert result in {"Infecté", "Sain"}