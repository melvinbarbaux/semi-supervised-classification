import numpy as np
from .base import SemiSupervisedMethod

class SelfTrainingMethod(SemiSupervisedMethod):
    """
    Implémentation de la méthode Self-Training :
      1. Entraîne le modèle sur X_l / y_l
      2. Prédit les probabilités sur X_u
      3. Sélectionne les instances au-dessus du seuil de confiance
      4. Ajoute ces instances au jeu étiqueté (avec leur pseudo-label)
      5. Répète jusqu'à max_iter ou plus de pseudo-labels trouvés
    """

    def run(self, X_l, y_l, X_u):
        # Copie des données pour itérer sans modifier l'entrée originale
        X_labeled = X_l.copy()
        y_labeled = y_l.copy()
        X_unlabeled = X_u.copy()

        for _ in range(self.max_iter):
            # Arrêt si plus d'exemples non étiquetés
            if X_unlabeled.shape[0] == 0:
                break

            # 1) entraînement sur les données labellisées
            self.model.train(X_labeled, y_labeled)

            # 2) prédiction de probabilités sur les non-étiquetées
            probs = self.model.predict_proba(X_unlabeled)
            confidences = probs.max(axis=1)
            pseudo_idx = np.where(confidences >= self.threshold)[0]

            # 3) arrêt si aucun confident
            if pseudo_idx.size == 0:
                break

            # 4) constitution des nouveaux points étiquetés
            new_X = X_unlabeled[pseudo_idx]
            new_y = probs[pseudo_idx].argmax(axis=1)

            # 5) mise à jour des jeux de données
            X_labeled = np.vstack([X_labeled, new_X])
            y_labeled = np.concatenate([y_labeled, new_y])
            # suppression des points ajoutés de X_unlabeled
            X_unlabeled = np.delete(X_unlabeled, pseudo_idx, axis=0)

        # Retourne le modèle entraîné et le jeu étiqueté final
        return self.model, X_labeled, y_labeled