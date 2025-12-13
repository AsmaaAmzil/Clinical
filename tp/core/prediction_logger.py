import json
import os
from datetime import datetime
from typing import Dict, Any, List


class PredictionLogger:
    """Système simple de journalisation des prédictions du modèle."""

    def __init__(self, log_dir: str = "logs"):
        """Initialise le répertoire de logs.

        Args:
            log_dir: Dossier où seront enregistrés les fichiers de log.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_prediction(
        self,
        patient_id: str,
        features: Dict[str, float],
        prediction: Any,
        confidence: float,
    ) -> str:
        """Enregistre une prédiction dans un fichier JSON.

        Args:
            patient_id: Identifiant du patient.
            features: Caractéristiques d'entrée utilisées pour la prédiction.
            prediction: Résultat prédit par le modèle.
            confidence: Score de confiance associé à la prédiction.

        Returns:
            Chemin du fichier de log créé.
        """
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "patient_id": patient_id,
            "features": features,
            "prediction": str(prediction),
            "confidence": float(confidence),
        }

        filename = f"prediction_{patient_id}_{int(datetime.now().timestamp())}.json"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

        return filepath

    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retourne l'historique des prédictions pour un patient donné."""
        if not os.path.isdir(self.log_dir):
            return []

        history: List[Dict[str, Any]] = []
        prefix = f"prediction_{patient_id}_"

        for name in os.listdir(self.log_dir):
            if not name.startswith(prefix):
                continue
            full_path = os.path.join(self.log_dir, name)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    history.append(data)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        return sorted(history, key=lambda x: x.get("timestamp", ""))
