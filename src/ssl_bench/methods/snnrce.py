import numpy as np
from copy import deepcopy
from typing import Tuple, Optional
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
import logging

from ssl_bench.methods.base import SemiSupervisedMethod
from ssl_bench.models.base import BaseModel

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)


class SnnrceMethod(SemiSupervisedMethod):
    """
    SNNRCE: Self-training Nearest Neighbor Rule using Cut Edges
    Adapté pour images 4D + NearestNeighbors sur version aplanie.
    """
    def __init__(
        self,
        base_model: BaseModel,
        n_neighbors: int = 10,
        alpha: float = 0.05,
        random_state: Optional[int] = None
    ):
        self.base_model = deepcopy(base_model)
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.random_state = random_state

    def run(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: np.ndarray
    ) -> Tuple[BaseModel, np.ndarray, np.ndarray]:
        logger.info("SNNRCE: démarrage")
        # Vues 4D pour le modèle, 2D pour NN
        L_x_4d = X_l.copy()
        U_x_4d = X_u.copy()
        L_x_flat = L_x_4d.reshape(len(L_x_4d), -1)
        U_x_flat = U_x_4d.reshape(len(U_x_4d), -1)
        logger.info(f"SNNRCE: Labled shape={L_x_4d.shape}, Unlabeled shape={U_x_4d.shape}")

        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y_l, return_counts=True)
        total_L = len(y_l)
        ratio = {c: counts[i] / total_L for i, c in enumerate(classes)}
        N_max = {c: int(ratio[c] * len(U_x_flat)) for c in classes}
        logger.info(f"SNNRCE: N_max par classe = {N_max}")

        L_y = y_l.copy()

        # Étape 1: bootstrap initial
        model = deepcopy(self.base_model)
        model.train(L_x_4d, L_y)
        logger.info("SNNRCE: entraînement initial effectué")

        # Étape 2: cut-edge initial
        nn0 = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn0.fit(L_x_flat)
        nbrs_U = nn0.kneighbors(U_x_flat, return_distance=False)[:,1:]
        mask0 = np.zeros(len(U_x_flat), dtype=bool)
        new0 = np.full(len(U_x_flat), -1, dtype=int)

        for i, nbrs in enumerate(nbrs_U):
            d = np.linalg.norm(U_x_flat[i] - L_x_flat[nbrs], axis=1)
            w = 1.0 / (1.0 + d)
            p = model.predict(U_x_4d[i:i+1])[0]
            R = w[(L_y[nbrs] != p)].sum() / (w.sum() + 1e-12)
            if R == 0.0:
                mask0[i] = True
                new0[i] = p

        added0 = mask0.sum()
        logger.info(f"SNNRCE: cut-edge initial—ajout de {added0} samples")
        if added0 > 0:
            L_x_4d = np.vstack([L_x_4d, U_x_4d[mask0]])
            L_y    = np.concatenate([L_y, new0[mask0]])
            U_x_4d = U_x_4d[~mask0]
            L_x_flat = L_x_4d.reshape(len(L_x_4d), -1)
            U_x_flat = U_x_4d.reshape(len(U_x_4d), -1)

        # Étape 3: self-training jusqu’à N_max
        iteration = 0
        while True:
            iteration += 1
            model = deepcopy(self.base_model)
            model.train(L_x_4d, L_y)
            if len(U_x_flat) == 0:
                logger.info("SNNRCE: plus d'exemples non-étiquetés, arrêt")
                break

            # confiance via distance NN
            nn1 = NearestNeighbors(n_neighbors=1)
            nn1.fit(L_x_flat)
            dists, _ = nn1.kneighbors(U_x_flat, return_distance=True)
            conf = np.exp(-dists.flatten())
            order = np.argsort(conf)[::-1]
            preds = model.predict(U_x_4d)

            to_take = []
            for c in classes:
                need = N_max[c] - np.sum(L_y == c)
                sel = [i for i in order if preds[i] == c][:max(0,need)]
                to_take.extend(sel)

            to_take = sorted(set(to_take))
            logger.info(f"SNNRCE: itération {iteration}, to_take={len(to_take)}")
            if not to_take:
                logger.info("SNNRCE: aucun nouvel ajout, arrêt self-training")
                break

            new_labels = preds[to_take]
            L_x_4d = np.vstack([L_x_4d, U_x_4d[to_take]])
            L_y    = np.concatenate([L_y, new_labels])
            mask = np.ones(len(U_x_flat), dtype=bool)
            mask[to_take] = False
            U_x_4d = U_x_4d[mask]
            L_x_flat = L_x_4d.reshape(len(L_x_4d), -1)
            U_x_flat = U_x_4d.reshape(len(U_x_4d), -1)

        # Étape 4: édition finale (cut-edge)
        nnL = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nnL.fit(L_x_flat)
        nbrs_L = nnL.kneighbors(L_x_flat, return_distance=False)[:,1:]
        Rv = []
        for i, nbrs in enumerate(nbrs_L):
            d = np.linalg.norm(L_x_flat[i] - L_x_flat[nbrs], axis=1)
            w = 1.0 / (1.0 + d)
            Rv.append(w[(L_y[nbrs] != L_y[i])].sum() / (w.sum() + 1e-12))
        Rv = np.array(Rv)
        mu, sigma = Rv.mean(), Rv.std()
        crit = mu + norm.ppf(1 - self.alpha/2) * sigma
        flips = 0
        for i, Ri in enumerate(Rv):
            if Ri > crit:
                L_y[i] = [c for c in classes if c != L_y[i]][0]
                flips += 1
        logger.info(f"SNNRCE: édition finale—flip de {flips} labels")

        # Étape 5: classification des restants
        if len(U_x_flat) > 0:
            final = deepcopy(self.base_model)
            final.train(L_x_4d, L_y)
            pU = final.predict(U_x_4d)
            L_x_4d = np.vstack([L_x_4d, U_x_4d])
            L_y    = np.concatenate([L_y, pU])
            logger.info(f"SNNRCE: classification finale de {len(pU)} restants")
        else:
            final = model

        logger.info(f"SNNRCE: terminé—total labelled = {len(L_y)}")
        return final, L_x_4d, L_y