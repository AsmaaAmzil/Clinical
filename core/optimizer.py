class Optimizer:
    """Optimisation des hyperparamètres"""
    
    def __init__(self, model_class, param_grid: dict):
        self.model_class = model_class
        self.param_grid = param_grid
        self.best_model = None
        self.best_params = None
    
    def optimize(self, X, y):
        """Recherche les meilleurs hyperparamètres"""
        from sklearn.model_selection import GridSearchCV
        
        base_model = self.model_class()
        grid_search = GridSearchCV(
            base_model.model, 
            self.param_grid, 
            cv=3, 
            scoring='accuracy'
        )
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        self.best_model = self.model_class(**self.best_params)
        self.best_model.train(X, y)
        
        return self.best_model, self.best_params