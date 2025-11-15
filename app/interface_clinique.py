class ClinicalPredictor:
    """
    Interface clinique pour la prédiction.
    Classe principale répondant à l'exercice.
    """
    
    def __init__(self, model: Model):
        """
        Initialise le prédicteur avec un modèle déjà entraîné.
        
        Args:
            model: Modèle IA pré-entraîné (instance de Model)
        """
        self.model = model
    
    def diagnose(self, patient_data):
        """
        Prédit le diagnostic pour un patient.
        
        Args:
            patient_data: Données du patient (array-like)
            
        Returns:
            "Infecté" si prédiction >= 0.5, "Sain" sinon
        """
        import numpy as np
        
        # S'assurer que les données sont au bon format
        if not isinstance(patient_data, np.ndarray):
            patient_data = np.array(patient_data)
        
        if len(patient_data.shape) == 1:
            patient_data = patient_data.reshape(1, -1)
        
        # Obtenir la probabilité de prédiction
        try:
            prediction_proba = self.model.predict_proba(patient_data)
            prediction_value = prediction_proba[0][1]  # Probabilité classe positive
        except:
            prediction = self.model.predict(patient_data)
            prediction_value = prediction[0]
        
        return "Infecté" if prediction_value >= 0.5 else "Sain"