class LogisticRegressionModel(Model):
    """Modèle de régression logistique"""
    
    def __init__(self, **kwargs):
        super().__init__()
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(**kwargs)
    
    def train(self, X, y):
        """Entraîne le modèle"""
        self.model.fit(X, y)
        self.is_trained = True
        return self