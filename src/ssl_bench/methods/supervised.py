from .base import SemiSupervisedMethod

class SupervisedMethod(SemiSupervisedMethod):
    """
    Baseline full-supervisé : ignore X_u et ne fait qu'un entraînement supervisé.
    """
    def run(self, X_l, y_l, X_u=None):
        # Entraînement standard sur le seul jeu étiqueté
        self.model.train(X_l, y_l)
        # On retourne le modèle et le jeu étiqueté inchangé
        return self.model, X_l, y_l