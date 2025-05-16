import logging
from copy import deepcopy
from typing import Tuple, Optional

import numpy as np
from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


class SupervisedMethod(SemiSupervisedMethod):
    def __init__(self, model: BaseModel, verbose: bool = False):
        super().__init__(model)
        self.verbose = verbose

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: Optional[np.ndarray] = None,
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        """
        :param X_l: labeled inputs, shape (n_samples, ...) 
        :param y_l: labels, shape (n_samples,)
        :param X_u: ignored
        :returns: (trained_model, X_l, y_l)
        """
        assert X_l.ndim in (2, 4)
        assert X_l.shape[0] == y_l.shape[0]
        assert X_l.shape[0] > 0

        if self.verbose:
            logger.info("Training on %d labeled samples", X_l.shape[0])
        model = deepcopy(self.model)
        model.train(X_l, y_l)
        return model, X_l, y_l