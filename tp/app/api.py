class ClinicalAPI:
    """API REST pour l'application clinique"""
    
    def __init__(self, predictor: ClinicalPredictor):
        self.predictor = predictor
    
    def predict_endpoint(self, patient_data: dict):
        """Endpoint de prédiction"""
        import numpy as np
        
        # Convertir les données du patient en array
        features = np.array(list(patient_data.values()))
        diagnosis = self.predictor.diagnose(features)
        
        return {
            "diagnosis": diagnosis,
            "confidence": "High" if diagnosis == "Infecté" else "Normal"
        }
