class Dataset:
    """Gestion des données d'entraînement et de test"""
    
    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_from_csv(self, filepath: str):
        """Charge les données depuis patient_data.csv"""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        df = pd.read_csv(filepath)
        X = df.drop('diagnosis', axis=1).values
        y = df['diagnosis'].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return self
    
    def get_train_data(self):
        """Retourne les données d'entraînement"""
        return self.X_train, self.y_train
    
    def get_test_data(self):
        """Retourne les données de test"""
        return self.X_test, self.y_test