from abc import ABC, abstractmethod

class SemiSupervisedMethod(ABC):
    """
    Interface pour une méthode d'apprentissage semi-supervisé.
    Chaque implémentation orchestrera l'utilisation d'un BaseModel
    pour entraîner sur les données étiquetées et exploiter les non-étiquetées.
    """

    def __init__(self, model, threshold: float = 0.8, max_iter: int = 10):
        """
        :param model: instance de BaseModel
        :param threshold: confiance minimale pour ajouter un pseudo-label
        :param max_iter: nombre maximum d'itérations
        """
        self.model = model
        self.threshold = threshold
        self.max_iter = max_iter

    @abstractmethod
    def run(self, X_l, y_l, X_u):
        """
        Lance la méthode semi-supervisée.
        :param X_l: array (n_labeled, n_features)
        :param y_l: array (n_labeled,)
        :param X_u: array (n_unlabeled, n_features)
        :returns: typiquement (trained_model, X_l_final, y_l_final)
        """
        pass