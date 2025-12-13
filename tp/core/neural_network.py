class NeuralNetworkModel(Model):
    """Modèle de réseau de neurones"""
    
    def __init__(self, hidden_layer_sizes=(100,), **kwargs):
        super().__init__()
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, **kwargs)
    
    def train(self, X, y):
        """Entraîne le modèle"""
        self.model.fit(X, y)
        self.is_trained = True
        return self