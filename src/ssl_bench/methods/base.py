from abc import ABC, abstractmethod

class SemiSupervisedMethod(ABC):
    """
    Interface for a semi-supervised learning method.
    Each implementation will orchestrate the use of a BaseModel
    to train on labeled data and leverage unlabeled data.
    """

    def __init__(self, model, threshold: float = 0.8, max_iter: int = 10):
        """
        :param model: an instance of BaseModel
        :param threshold: minimum confidence required to add a pseudo-label
        :param max_iter: maximum number of iterations
        """
        self.model = model
        self.threshold = threshold
        self.max_iter = max_iter

    @abstractmethod
    def run(self, X_l, y_l, X_u):
        """
        Execute the semi-supervised method.
        
        :param X_l: numpy array of shape (n_labeled, n_features), labeled inputs
        :param y_l: numpy array of shape (n_labeled,), labeled targets
        :param X_u: numpy array of shape (n_unlabeled, n_features), unlabeled inputs
        :returns: typically a tuple (trained_model, X_l_final, y_l_final)
        """
        pass