class Trainer:
    """Gestion de l'entraînement du modèle"""
    
    def __init__(self, model: Model, dataset: Dataset):
        self.model = model
        self.dataset = dataset
        self.training_history = {}
    
    def train(self):
        """Entraîne le modèle sur le dataset"""
        X_train, y_train = self.dataset.get_train_data()
        self.model.train(X_train, y_train)
        return self
    
    def get_trained_model(self):
        """Retourne le modèle entraîné"""
        return self.model