class Model:
    """Interface de base pour tous les modèles"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def predict(self, X):
        """Fait une prédiction"""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de prédire")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Retourne les probabilités de prédiction"""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de prédire")
        return self.model.predict_proba(X)