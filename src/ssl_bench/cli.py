# src/ssl_bench/cli.py

import json
import click
import pandas as pd
from copy import deepcopy

from ssl_bench.cache import CacheManager
from ssl_bench.experiment import ExperimentRunner, ExperimentConfig, ExperimentResult
from ssl_bench.registry import ModelRegistry
from ssl_bench.models.sklearn import RandomForestWrapper
from ssl_bench.transforms import DataTransformer

from ssl_bench.methods.supervised import SupervisedMethod
from ssl_bench.methods.self_training import SelfTrainingMethod
from ssl_bench.methods.setred import SetredMethod
from ssl_bench.methods.snnrce import SnnrceMethod
from ssl_bench.methods.tri_training import TriTrainingMethod
from ssl_bench.methods.democratic_co_learning import DemocraticCoLearningMethod


@click.group()
def cli():
    """SSL-Bench CLI: lancez, cachez, et prédisez vos benchmarks SSL."""
    pass


@cli.command()
@click.option('--method',
    type=click.Choice([
        'supervised',
        'self_training',
        'setred',
        'snnrce',
        'tri_training',
        'democratic_co_learning'
    ]),
    required=True,
    help="Méthode SSL à utiliser.")
@click.option('--model',        type=str,   default='RandomForest')
@click.option('--dataset',      type=str,   default='Synthetic')
@click.option('--labeled-fraction', type=float, default=0.3)
@click.option('--seed',         type=int,   default=0)
@click.option('--n-estimators', type=int,   default=10)
@click.option('--threshold',    type=float, default=0.8)
@click.option('--max-iter',     type=int,   default=5)
@click.option('--theta',        type=float, default=0.1)
@click.option('--pool-size',    type=int,   default=None)
@click.option('--n-neighbors',  type=int,   default=10)
@click.option('--alpha',        type=float, default=0.05)
@click.option('--cache-dir',    type=click.Path(), default='data/processed/experiment_cache')
@click.option('--registry-dir', type=click.Path(), default='data/processed/registry')
def run(
    method, model, dataset, labeled_fraction, seed,
    n_estimators, threshold, max_iter,
    theta, pool_size, n_neighbors, alpha,
    cache_dir, registry_dir
):
    """
    Lance une unique expérience SSL et affiche les métriques.
    """
    # 1) Config
    config = ExperimentConfig(
        method=method,
        model=model,
        dataset=dataset,
        labeled_fraction=labeled_fraction,
        seed=seed,
        model_hyperparams={'n_estimators': n_estimators},
        method_hyperparams={
            'threshold': threshold,
            'max_iter': max_iter,
            'theta': theta,
            'pool_size': pool_size,
            'n_neighbors': n_neighbors,
            'alpha': alpha
        }
    )

    # 2) Instanciation
    cache    = CacheManager(cache_dir=cache_dir)
    runner   = ExperimentRunner(cache)
    registry = ModelRegistry(registry_dir=registry_dir)

    # 3) train/eval closure
    def train_eval(cfg):
        X, y = pd.DataFrame()  # placeholder
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=200, n_features=20,
            n_informative=10, n_redundant=5,
            n_classes=2, random_state=cfg.seed
        )
        # preprocessing
        transformer = DataTransformer(test_size=1-cfg.labeled_fraction, seed=cfg.seed)
        Xs, ys = transformer.fit_transform(X, y)
        X_train, X_test, y_train_enc, y_test_enc = transformer.split(Xs, ys)
        y_train = y_train_enc.argmax(axis=1)
        y_test  = y_test_enc.argmax(axis=1)

        # base model
        rf = RandomForestWrapper(n_estimators=cfg.model_hyperparams['n_estimators'])

        # select method
        if cfg.method == 'supervised':
            method_obj = SupervisedMethod(rf)
        elif cfg.method == 'self_training':
            method_obj = SelfTrainingMethod(
                rf,
                threshold=cfg.method_hyperparams['threshold'],
                max_iter=cfg.method_hyperparams['max_iter']
            )
        elif cfg.method == 'setred':
            method_obj = SetredMethod(
                rf,
                theta=cfg.method_hyperparams['theta'],
                max_iter=cfg.method_hyperparams['max_iter'],
                pool_size=cfg.method_hyperparams['pool_size'],
                n_neighbors=cfg.method_hyperparams['n_neighbors'],
                random_state=cfg.seed
            )
        elif cfg.method == 'snnrce':
            method_obj = SnnrceMethod(
                rf,
                n_neighbors=cfg.method_hyperparams['n_neighbors'],
                alpha=cfg.method_hyperparams['alpha'],
                random_state=cfg.seed
            )
        elif cfg.method == 'tri_training':
            method_obj = TriTrainingMethod(rf)
        elif cfg.method == 'democratic_co_learning':
            learners = [deepcopy(rf) for _ in range(3)]
            method_obj = DemocraticCoLearningMethod(
                learners=learners,
                alpha=cfg.method_hyperparams['alpha'],
                random_state=cfg.seed
            )
        else:
            raise click.ClickException(f"Method {cfg.method} inconnue.")

        # train + pseudo-label
        final_model, X_lab, y_lab = method_obj.run(X_train, y_train, X_test)
        preds = final_model.predict(X_test)
        acc   = (preds == y_test).mean()

        return final_model, {
            'accuracy': acc,
            'n_labeled': X_lab.shape[0]
        }

    # 4) Exécute et cache
    result: ExperimentResult = runner.run(config, train_eval)

    # 5) Registry
    registry.register_run(
        dataset=config.dataset,
        model_name=config.model,
        method=config.method,
        trained_model=result.model,
        metrics=result.metrics
    )

    # 6) Affichage
    click.echo(f"Method: {config.method}")
    click.echo(f"Dataset: {config.dataset}")
    click.echo(f"Accuracy: {result.metrics['accuracy']:.4f}")
    click.echo(f"# labeled: {result.metrics['n_labeled']}")
    click.echo(f"Duration: {result.duration:.3f}s")


@cli.command()
@click.option('--config-json', type=click.Path(exists=True), required=True)
@click.option('--input-csv',  type=click.Path(exists=True), required=True)
@click.option('--output-csv', type=click.Path(), default='predictions.csv')
@click.option('--registry-dir', type=click.Path(), default='data/processed/registry')
def predict(config_json, input_csv, output_csv, registry_dir):
    """
    Charge le modèle best depuis registry et prédit sur X_new.
    """
    with open(config_json) as f:
        cfg = ExperimentConfig(**json.load(f))

    registry = ModelRegistry(registry_dir=registry_dir)
    model    = registry.get_best_model(
        dataset=cfg.dataset,
        model_name=cfg.model,
        method=cfg.method
    )
    if model is None:
        raise click.ClickException("Aucun modèle best trouvé.")

    df = pd.read_csv(input_csv)
    X_new = df.values
    preds = model.predict(X_new)

    pd.DataFrame({'prediction': preds}).to_csv(output_csv, index=False)
    click.echo(f"Predictions saved to {output_csv}")


if __name__ == '__main__':
    cli()