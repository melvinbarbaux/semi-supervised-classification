# src/ssl_bench/methods/__init__.py

from .base import SemiSupervisedMethod
from .supervised import SupervisedMethod
from .self_training import SelfTrainingMethod
from .setred import SetredMethod
from .tri_training import TriTrainingMethod
from .democratic_co_learning import DemocraticCoLearningMethod
from .dash import DashMethod
from .mssboost import MSSBoostMethod
from .poisson_learning import PoissonLearningMethod
from .gfhf import GFHFMethod
from .ebsa import EBSAMethod
from .ttadec import TTADECMethod


__all__ = [
    "SemiSupervisedMethod",
    "SupervisedMethod",
    "SelfTrainingMethod",
    "SetredMethod",
    "TriTrainingMethod",
    "DemocraticCoLearningMethod",
    "DashMethod",
    "MSSBoostMethod",
    "EBSAMethod",
    "TTADECMethod"
    "GFHFMethod",
    "PoissonLearningMethod",
]