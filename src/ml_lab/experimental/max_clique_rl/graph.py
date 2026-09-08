from __future__ import annotations

import numpy as np


def validate_adjacency(graph) -> np.ndarray:
    matrix = np.asarray(graph, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("graph must be a square adjacency matrix")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("graph adjacency matrix must be symmetric")
    if np.any((matrix != 0) & (matrix != 1)):
        raise ValueError("graph adjacency matrix must contain only 0/1 values")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    return matrix


def generate_planted_clique_graph(
    num_nodes: int = 30,
    clique_size: int = 8,
    *,
    background_edge_probability: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if clique_size <= 0 or clique_size > num_nodes:
        raise ValueError("clique_size must be between 1 and num_nodes")
    if not 0.0 <= background_edge_probability <= 1.0:
        raise ValueError("background_edge_probability must be in [0, 1]")

    rng = np.random.default_rng(seed)
    graph = np.zeros((num_nodes, num_nodes), dtype=np.int8)
    upper = rng.random((num_nodes, num_nodes)) < background_edge_probability
    upper = np.triu(upper, 1)
    graph[upper] = 1
    graph = graph + graph.T
    planted = list(range(clique_size))
    for i in planted:
        for j in planted:
            if i != j:
                graph[i, j] = 1
    return graph, planted


def is_clique(graph, nodes) -> bool:
    matrix = validate_adjacency(graph)
    members = [int(node) for node in nodes]
    if len(set(members)) != len(members):
        return False
    if any(node < 0 or node >= matrix.shape[0] for node in members):
        return False
    return all(matrix[i, j] == 1 for pos, i in enumerate(members) for j in members[pos + 1 :])


def valid_additions(graph, clique) -> list[int]:
    matrix = validate_adjacency(graph)
    current = [int(node) for node in clique]
    return [
        node
        for node in range(matrix.shape[0])
        if node not in current and all(matrix[node, member] == 1 for member in current)
    ]


def greedy_clique(graph, *, start_node: int | None = None) -> list[int]:
    """Simple deterministic clique constructor used as a non-neural baseline."""
    matrix = validate_adjacency(graph)
    degrees = matrix.sum(axis=1)
    if start_node is None:
        start_node = int(np.argmax(degrees))
    if start_node < 0 or start_node >= matrix.shape[0]:
        raise ValueError("start_node is outside the graph")

    clique = [int(start_node)]
    while True:
        candidates = valid_additions(matrix, clique)
        if not candidates:
            break
        best = max(candidates, key=lambda node: (int(degrees[node]), -int(node)))
        clique.append(int(best))
    return clique
