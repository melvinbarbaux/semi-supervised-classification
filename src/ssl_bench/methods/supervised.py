import logging
from copy import deepcopy
from typing import Tuple, Optional

import numpy as np

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(ch)


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
        # 1) Vérifications basiques
        assert X_l.ndim == 2 or X_l.ndim == 4, "X_l doit être 2D (tabulaire) ou 4D (images)"
        assert X_l.shape[0] == y_l.shape[0], "X_l et y_l doivent avoir le même nombre d'échantillons"
        assert X_l.shape[0] > 0, "Aucun échantillon labellisé fourni"

        # 2) Entraînement
        if self.verbose:
            logger.info("SupervisedMethod: training on %d samples", X_l.shape[0])
        model = deepcopy(self.model)
        model.train(X_l, y_l)
        if self.verbose:
            logger.info("SupervisedMethod: training completed")

        # 3) On renvoie le modèle entraîné et le jeu labellisé
        return model, X_l, y_l