class Preprocessor:
    """Prétraitement des données"""
    
    def __init__(self):
        self.scaler = None
    
    def normalize(self, X):
        """Normalise les données"""
        from sklearn.preprocessing import StandardScaler
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def handle_missing_values(self, X):
        """Gère les valeurs manquantes"""
        import numpy as np
        from sklearn.impute import SimpleImputer
        
        imputer = SimpleImputer(strategy='mean')
        return imputer.fit_transform(X)
