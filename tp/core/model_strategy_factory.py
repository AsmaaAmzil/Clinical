from core.model import Model
from core.logistic_regression import LogisticRegressionModel
from core.neural_network import NeuralNetworkModel


class ModelStrategyFactory:
    """Factory/Strategy selector pour choisir dynamiquement le modèle IA.

    Le client choisit une "stratégie" (logistic, neural, ...) et cette
    classe retourne une instance concrète de Model correspondante.
    """

    @staticmethod
    def create_model(strategy: str, **kwargs) -> Model:
        """Crée un modèle en fonction de la stratégie demandée.

        Args:
            strategy: Nom de la stratégie ("logistic", "neural", ...).
            **kwargs: Paramètres passés au constructeur du modèle concret.

        Returns:
            Instance de Model correspondant à la stratégie choisie.

        Raises:
            ValueError: si la stratégie est inconnue.
        """
        name = strategy.lower()

        if name == "logistic":
            return LogisticRegressionModel(**kwargs)
        if name == "neural":
            return NeuralNetworkModel(**kwargs)

        raise ValueError(f"Unknown model strategy: {strategy}")
