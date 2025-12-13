from core.model import Model
from core.dataset import Dataset


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


class TrainerBuilder:
    """Builder pour configurer et créer un Trainer.

    Permet de chaîner les appels pour construire un Trainer, avec
    option d'entraînement automatique dans build().
    """

    def __init__(self):
        self._model: Model | None = None
        self._dataset: Dataset | None = None
        self._auto_train: bool = False

    def with_model(self, model: Model) -> "TrainerBuilder":
        """Spécifie le modèle à entraîner."""
        self._model = model
        return self

    def with_dataset(self, dataset: Dataset) -> "TrainerBuilder":
        """Spécifie le dataset à utiliser pour l'entraînement."""
        self._dataset = dataset
        return self

    def auto_train(self, enabled: bool = True) -> "TrainerBuilder":
        """Active ou désactive l'entraînement automatique dans build()."""
        self._auto_train = enabled
        return self

    def build(self) -> Trainer:
        """Crée un Trainer (et lance l'entraînement si auto_train est activé)."""
        if self._model is None:
            raise ValueError("TrainerBuilder: model must be provided")
        if self._dataset is None:
            raise ValueError("TrainerBuilder: dataset must be provided")

        trainer = Trainer(self._model, self._dataset)

        if self._auto_train:
            trainer.train()

        return trainer