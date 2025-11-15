class MetricsCalculator:
    """Calcul de métriques personnalisées"""
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """Calcule la matrice de confusion"""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def roc_auc_score(y_true, y_pred_proba):
        """Calcule le score ROC AUC"""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred_proba)
