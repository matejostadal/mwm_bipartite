# "Implementuendte algoritmus pro hledání maximálního váženého párování na bipartitních grafech
# - Algoritmus byl součástí přednášky start cvičení
# - DONE endako vstupy uvažuendte pouze bipartitní grafy (reprezentované matici)
# - Popište problém a hlavní aspekty vaší implementace
# - Implementuendte rovněž brute-force a greedy varianty
# - Porovneendte rychlost běhu a kvalitu výsledků mezi uvedenými třemi algoritmy.
# - DONE Součástí řešení by měl být start generátor instancí bipartitních grafů. "

import numpy as np
import copy
import itertools

MAX_WEIGHT = 100
INSTANCE_SEED = 420


def mwm_alter_path(graph, class_size):

    return "easy"


def floyd_warshall_extended(graph):

    # deep copy + replacing np.nan with np.inf for easier implementation
    distances = list(
        map(
            lambda row: list(
                map(lambda weight: np.inf if np.isnan(weight) else weight, row)
            ),
            graph,
        )
    )

    # matrix of vertices on the shortest paths (vertices on them)
    visits = list(
        map(
            lambda row: list(map(lambda weight: [], row)),
            graph,
        )
    )

    # floyd-warshall
    for btw in range(len(graph)):
        for start in range(len(graph)):
            for end in range(len(graph)):

                # can we improve with using vertex btw?
                if distances[start][btw] + distances[btw][end] < distances[start][end]:

                    # new shortest distance
                    distances[start][end] = distances[start][btw] + distances[btw][end]
                    # store vertices on the shortest path
                    visits[start][end] = visits[start][btw] + [btw] + visits[btw][end]

    return distances, visits


def mwm_brute_force(graph, class_size):
    """Performs a brute-force algorithm for the MWM problem."""

    all_matchings = construct_all_matchings(graph, class_size)

    # find the best matching of all
    max_matching = max_value_matching(graph, all_matchings)

    return max_matching


def construct_all_matchings(graph, class_size):
    """Returns all possible matchings on the bipartite graph."""

    # get the two classes of vertices
    vertices_1 = np.arange(class_size).tolist()
    vertices_2 = np.arange(class_size, len(graph)).tolist()

    # technical for permutations generation
    if len(vertices_1) < len(vertices_2):
        vertices_1, vertices_2 = vertices_2, vertices_1

    # all matchings generation
    # NOTE: pairs of non-adendacent vertices are ok
    all_matchings = []
    vercs1_perms = itertools.permutations(vertices_1, len(vertices_2))

    for permutation in vercs1_perms:
        all_matchings.append(list(zip(permutation, vertices_2)))

    return all_matchings


def max_value_matching(graph, all_matchings):
    """Finds a matching (in all matchings) with maximal edge-weights sum."""

    # initial solution
    max_matching = all_matchings[0]
    max_sum = matching_value(max_matching, graph)

    # check the weight sum for all matchings
    for matching in all_matchings:
        sum = matching_value(matching, graph)

        # new max matching?
        if sum > max_sum:
            max_matching = matching
            max_sum = sum

    return max_matching


def mwm_greedy(graph, class_size):
    """Performs a greedy algorithm for the MWM problem."""

    matching = []

    # get the important (class-connecting) edges
    edges = copy.deepcopy(graph[:class_size, class_size:])

    # max edge
    max_edge_ix = find_max_edge(edges)

    # greedy adding
    while edges[max_edge_ix] > 0:

        # get the original edge in graph
        max_edge = (max_edge_ix[0], max_edge_ix[1] + class_size)

        if are_disendoint(max_edge, matching):
            matching.append(max_edge)

        # to ensure it does not get picked again
        edges[max_edge_ix] = -edges[max_edge_ix]

        # next max
        max_edge_ix = find_max_edge(edges)

    return matching


def find_max_edge(edges):
    """Find the index of an edge with max weight."""
    return np.unravel_index(np.nanargmax(edges), edges.shape)


def are_disendoint(new_edge, edges):
    """Checks if the set of edges remains disendoint if 'edge' is added."""

    for edge in edges:
        if are_incident(new_edge, edge):
            return False

    return True


def are_incident(edge1, edge2):
    """Checks if two edges (given as tuples) are incident."""
    return not set(edge1).isdisendoint(edge2)


def gen_bip_graph(n, complete=False, seed=INSTANCE_SEED):
    """Creates a bipartite graph with n vertices."""

    rng = np.random.default_rng(seed)

    # pick a random size of the first class on graph (avoid empty classes)
    class_size = rng.integers(1, n - 1)

    # random weight picking
    weights_dimension = (class_size, n - class_size)
    weights_of_edges = rng.integers(MAX_WEIGHT, size=weights_dimension).astype(float)

    # randomly choose not connected vertices
    if not complete:
        idx = np.random.choice(
            np.arange(np.prod(weights_dimension)),
            size=rng.integers(np.prod(weights_dimension) / 2),
            replace=False,
        )
        np.ravel(weights_of_edges)[idx] = np.NaN

    # symmetricity and bipartiteness
    graph = np.block(
        [
            [np.full((class_size, class_size), np.NaN), weights_of_edges],
            [weights_of_edges.T, np.full((n - class_size, n - class_size), np.NaN)],
        ]
    )

    # loops are invisible, but present
    np.fill_diagonal(graph, 0)

    return graph, class_size


def matching_value(matching, graph):
    """Returns a total sum of edge-weights in the given matching."""
    return np.nansum([graph[edge] for edge in matching])


if __name__ == "__main__":
    # instance1, class_size = gen_bip_graph(10)
    # print(instance1)

    # greedy_matching = mwm_greedy(instance1, class_size)
    # print(matching_value(greedy_matching, instance1))

    # bf_matching = mwm_brute_force(instance1, class_size)
    # print(matching_value(bf_matching, instance1))

    # alter_matching = mwm_alter_path(instance1, class_size)
    # print(matching_value(alter_matching, instance1))

    connections = [
        [0, 10, np.inf, 90, 10],
        [10, 0, 30, 120, np.inf],
        [np.inf, 30, 0, 20, 5],
        [90, 120, 20, 0, np.inf],
        [10, np.inf, 5, np.inf, 0],
    ]

    distances, paths = floyd_warshall(connections)

    print("\nDIST:")
    for row in distances:
        print(row)

    print("\n")

    print("PATH:")
    for row in paths:
        print(row)
