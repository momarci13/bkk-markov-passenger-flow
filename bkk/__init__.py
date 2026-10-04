"""
bkk – Multi-modal Markov-chain passenger-flow model
for the BKK Budapest GTFS network.

Public API
----------
GTFSLoader          – download and parse the BKK GTFS ZIP
NetworkBuilder      – build sparse modal subgraphs + cost matrices
GeneratorBuilder    – assemble CTMC generator matrices Q^v (CSR)
DemandPrior         – E1 / E2 / E3 prior estimators for N_i(0)
KFESolver           – Kolmogorov Forward Equation (ODE / matrix-exp)
GillespieSSA        – exact Doob-Gillespie direct method
TauLeap             – tau-leaping approximation
ResilienceAnalyser  – Kemeny / spectral-gap / efficiency vulnerability
build_line_network  – hubs + line segments for one service window (model v3)
OpenNetworkModel    – line-aware open Markov network, exact Poisson solution
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bkk-markov-flow")
except PackageNotFoundError:
    __version__ = "3.0.0-dev"

from .gtfs      import GTFSLoader
from .network   import NetworkBuilder
from .generator import GeneratorBuilder
from .demand    import DemandPrior
from .simulate  import KFESolver, GillespieSSA, TauLeap, allocate_modal_population
from .resilience import ResilienceAnalyser
from .linemodel import LineNetwork, ModelParams, OpenNetworkModel, build_line_network

__all__ = [
    "GTFSLoader",
    "NetworkBuilder",
    "GeneratorBuilder",
    "DemandPrior",
    "KFESolver",
    "GillespieSSA",
    "TauLeap",
    "allocate_modal_population",
    "ResilienceAnalyser",
    "LineNetwork",
    "ModelParams",
    "OpenNetworkModel",
    "build_line_network",
]
