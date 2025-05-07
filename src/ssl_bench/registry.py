import os
import json
import joblib
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

class ModelRegistry:
    """
    Gère l'enregistrement structuré des runs et la sélection du meilleur modèle.
    """
    def __init__(self, registry_dir: str = "data/processed/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, dataset: str, model_name: str, method: str) -> Path:
        return self.registry_dir / dataset / model_name / method

    def register_run(
        self,
        dataset: str,
        model_name: str,
        method: str,
        trained_model: Any,
        metrics: dict,
        replace_best: bool = True
    ) -> str:
        """
        Enregistre un nouveau run et met à jour le best si nécessaire.
        :returns: run_id
        """
        base = self._path(dataset, model_name, method)
        runs_dir = base / "runs"
        best_dir = base / "best"
        runs_dir.mkdir(parents=True, exist_ok=True)
        best_dir.mkdir(parents=True, exist_ok=True)

        # Génération d'un ID unique pour le run
        run_id = uuid.uuid4().hex
        run_dir = runs_dir / run_id
        run_dir.mkdir()

        # Sauvegarde du modèle
        model_path = run_dir / "model.pkl"
        joblib.dump(trained_model, model_path)
        # Sauvegarde des metrics
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        # Vérifier si on doit remplacer le best
        if replace_best:
            best_metrics_path = best_dir / "metrics_best.json"
            if not best_metrics_path.exists():
                # Pas encore de best
                self._copy_run_as_best(run_dir, best_dir)
            else:
                with best_metrics_path.open() as f:
                    best_metrics = json.load(f)
                # Comparaison sur accuracy (plus haut est meilleur)
                if metrics.get("accuracy", 0) > best_metrics.get("accuracy", 0):
                    self._copy_run_as_best(run_dir, best_dir)
        return run_id

    def _copy_run_as_best(self, run_dir: Path, best_dir: Path) -> None:
        # Copie model.pkl et metrics.json vers best/
        src_model = run_dir / "model.pkl"
        src_metrics = run_dir / "metrics.json"
        dst_model = best_dir / "model_best.pkl"
        dst_metrics = best_dir / "metrics_best.json"
        shutil.copy2(src_model, dst_model)
        shutil.copy2(src_metrics, dst_metrics)

    def get_best_model(self, dataset: str, model_name: str, method: str) -> Optional[Any]:
        """
        Charge et retourne le meilleur modèle pour la combinaison.
        """
        best_model = self._path(dataset, model_name, method) / "best" / "model_best.pkl"
        if best_model.exists():
            return joblib.load(best_model)
        return None

    def get_best_metrics(self, dataset: str, model_name: str, method: str) -> Optional[dict]:
        """
        Retourne les metrics du meilleur run.
        """
        best_metrics = self._path(dataset, model_name, method) / "best" / "metrics_best.json"
        if best_metrics.exists():
            with best_metrics.open() as f:
                return json.load(f)
        return None

    def list_runs(
        self,
        dataset: str,
        model_name: str,
        method: str
    ) -> list[str]:
        """
        Liste des run_id enregistrés pour la combinaison donnée.
        """
        runs_dir = self._path(dataset, model_name, method) / "runs"
        if not runs_dir.exists():
            return []
        return [p.name for p in runs_dir.iterdir() if p.is_dir()]

    def resume_run(
        self,
        dataset: str,
        model_name: str,
        method: str,
        run_id: str
    ) -> Any:
        """
        Recharge le modèle d'un run existant pour reprendre l'entraînement.
        """
        run_dir = self._path(dataset, model_name, method) / "runs" / run_id
        model_path = run_dir / "model.pkl"
        if model_path.exists():
            return joblib.load(model_path)
        raise FileNotFoundError(f"Run {run_id} not found.")