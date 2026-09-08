"""Experimental maximum-clique reinforcement-learning sandbox.

The package isolates graph-specific RL code recovered from the legacy Max Clique
ML Solver.  It does not establish a stable reinforcement-learning API.
"""

from .graph import generate_planted_clique_graph, is_clique, greedy_clique
from .pipeline import run

__all__ = ["generate_planted_clique_graph", "greedy_clique", "is_clique", "run"]
