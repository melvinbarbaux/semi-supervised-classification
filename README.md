# SSL-Bench

**SSL-Bench** est un framework Python de benchmarking pour la classification semi-supervisée. Il permet de comparer facilement différentes méthodes (baseline supervisées et approches semi-supervisées) sur divers datasets (images, texte, tabulaire, etc.), de gérer le cache des expériences, de réimporter les meilleurs modèles et de lancer des workflows complets via CLI ou scripts.

---

## Table des matières

1. [Caractéristiques principales](#caractéristiques-principales)
2. [Installation](#installation)
3. [Structure du projet](#structure-du-projet)
4. [Utilisation rapide](#utilisation-rapide)

   * [Run minimal](#run-minimal)
   * [Boucle d'expériences](#boucle-dexpériences)
   * [Benchmark complet](#benchmark-complet)
   * [CLI](#cli)
5. [Modules](#modules)
6. [Tests](#tests)
7. [Contribuer](#contribuer)
8. [Licence](#licence)

---

## Caractéristiques principales

* Chargement unifié de datasets (images, texte, tabulaire) via `DatasetLoader`.
* Pipelines de pré-traitement cohérents avec `DataTransformer` (scaling, encodage, split étiqueté/non-étiqueté).
* Wrappers pour modèles scikit-learn (`RandomForestWrapper`, `SVMWrapper`) et PyTorch (`TorchModel`).
* Méthodes semi-supervisées modulaires : Self-Training, baselines supervisées.
* Orchestrateur d'expériences `ExperimentRunner` avec cache Joblib pour éviter les recomputations.
* Registry structuré des runs et gestion automatique du *best* modèle.
* Interface CLI (*Click*) pour lancer un run unique (`ssl-bench run`) ou prédire (`ssl-bench predict`).
* Scripts d'exemple : `run_minimal.py`, `experiment_loop.py`, `run_full_benchmark.py`.
* Couverture de tests complète avec pytest.

---

## Installation

1. Cloner le dépôt :

   ```bash
   git clone https://github.com/username/semi-supervised-classification.git
   cd semi-supervised-classification
   ```

2. Installer avec Poetry (Python >= 3.12) :

   ```bash
   poetry install
   poetry add click matplotlib  # si besoin de cli + plots
   ```

3. (Optionnel) Exposer la commande `ssl-bench` :

   ```toml
   # pyproject.toml
   [tool.poetry.scripts]
   ssl-bench = "ssl_bench.cli:cli"
   ```

   puis : `poetry install`

---

## Structure du projet

```
├── README.md
├── docs/               → Documentation supplémentaires
├── data/
│   ├── raw/            → jeux de données brutes (CIFAR‑10, etc.)
│   └── processed/      → cache des expériences, registry
├── examples/
│   ├── run_minimal.py
│   ├── experiment_loop.py
│   └── run_full_benchmark.py
├── poetry.lock
├── pyproject.toml
├── src/
│   └── ssl_bench/
│       ├── __init__.py
│       ├── benchmark.py
│       ├── cache.py
│       ├── cli.py
│       ├── datamodule/
│       │   ├── base.py
│       │   ├── image.py
│       │   ├── tabular.py
│       │   └── text.py
│       ├── experiment.py
│       ├── methods/
│       │   ├── base.py
│       │   ├── self_training.py
│       │   └── supervised.py
│       ├── models/
│       │   ├── base.py
│       │   ├── sklearn.py
│       │   └── torch.py
│       ├── predictor.py
│       ├── registry.py
│       └── transforms/
│           ├── __init__.py
│           └── transformer.py
└── tests/
    ├── fixtures/
    ├── test_cifar10_raw.py
    ├── test_data_module.py
    ├── test_dataset_loader.py
    ├── test_experiment.py
    ├── test_methods.py
    ├── test_methods_supervised.py
    ├── test_models.py
    ├── test_predictor.py
    ├── test_smoke.py
    └── test_transformer.py
```

---

## Utilisation rapide

### Run minimal

```bash
poetry run python examples/run_minimal.py
```

### Boucle d'expériences (Grid Search)

```bash
poetry run python examples/experiment_loop.py
```

### Benchmark complet sur CIFAR-10

```bash
# Assurez-vous d'avoir téléchargé les batches dans data/raw/cifar10/cifar-10-batches-py/
poetry run python examples/run_full_benchmark.py
```

### CLI

```bash
# Run unique via CLI
poetry run python -m ssl_bench.cli run \
  --method self_training \
  --model RandomForest \
  --dataset Synthetic \
  --labeled-fraction 0.3 \
  --seed 0 \
  --n-estimators 20 \
  --threshold 0.8 \
  --max-iter 5

# Predict à partir d'un modèle mis en cache
poetry run python -m ssl_bench.cli predict \
  --config-json cfg.json \
  --input-csv new_samples.csv \
  --output-csv preds.csv
```

---

## Modules clés

* **`datamodule/`** : loaders et pré-traitements des jeux de données
* **`models/`** : wrappers `scikit-learn` + `PyTorch` (init automatique GPU/CPU)
* **`methods/`** : algorithmes SSL (Self-Training, Supervised)
* **`experiment.py`** : orchestration + cache via `ExperimentRunner`
* **`registry.py`** : stockage structuré des runs et best-model
* **`cli.py`** : interface en ligne de commande
* **`benchmark.py`** : `BenchmarkSuite` pour lancer des grilles systématiques

---

## Tests

Lancez tous les tests avec pytest :

```bash
poetry run pytest -q
```

---

## Contribuer

1. Forkez le projet
2. Créez une branche : `git checkout -b feat/awesome`
3. Faites vos modifications / ajoutez des tests
4. Ouvrez une Pull Request

Merci pour vos contributions !

---

## Licence

MIT © Melvin
