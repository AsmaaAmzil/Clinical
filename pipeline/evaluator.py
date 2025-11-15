class Evaluator:
    """Évaluation des performances du modèle"""
    
    def __init__(self, model: Model):
        self.model = model
        self.metrics = {}
    
    def evaluate(self, X_test, y_test):
        """Évalue le modèle sur les données de test"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.model.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1_score': f1_score(y_test, predictions, zero_division=0)
        }
        return self.metrics
    
    def print_report(self):
        """Affiche un rapport des métriques"""
        print("\n📊 RAPPORT D'ÉVALUATION")
        print("-" * 40)
        for metric, value in self.metrics.items():
            print(f"  {metric.capitalize():12s}: {value:.2%}")