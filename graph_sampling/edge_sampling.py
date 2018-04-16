"""
Edge and pair sampling
"""

import numpy as np

from helpers.graph_helpers import edge_list_to_matrix
from helpers.sampling_helpers import choice


def edge_samp(graph, k, replace=False, return_split=False):
    """
    sample a subgraph by choosing k edges uniformly at random
    :param graph: (nparray, nparray) of (edge_list, edge_weights)
    :param k: int
    :param replace: bool
    :return:
    """
    edge_list, weights = graph

    num_edges = edge_list.shape[0]
    select = choice(num_edges, k, replace=replace)
    selected_edges = np.copy(edge_list[select])
    selected_weights = np.copy(weights[select])

    if not return_split:
        return (selected_edges, selected_weights)
    else:
        not_selected = np.in1d(range(num_edges), select, assume_unique=True, invert=True)
        not_selected_edges = np.copy(edge_list[not_selected])
        not_selected_weights = np.copy(weights[not_selected])
        return (selected_edges, selected_weights), (not_selected_edges, not_selected_weights)


def edge_samp_multi(graph, k, replace=False):
    """
    sample a subgraph by choosing k edges uniformly at random
    :param graph: (nparray, nparray) of (edge_list, edge_weights). Edge weights must be natural numbers!
    :param k: int
    :param replace: bool
    :return:
    """
    edge_list, weights = graph

    cw = np.cumsum(weights)

    # equivalent to selecting uniformly at random from all edges (including replicates)
    num_edges = cw[-1]
    selected_edges = choice(num_edges, k, replace=replace)
    select = np.searchsorted(cw, selected_edges)

    select_unique, selected_weights = np.unique(select, return_counts=True)

    selected_edges = np.copy(edge_list[select_unique])

    return (selected_edges, selected_weights)


def edge_sample_generator(input_graph, k, replace=False, multi=False):
    """
    generates uniform edge subsamples from input_graph, returned as adjacency matrices.
    Each sample also returns arrays giving the identities of the selected vertices and items in input_graph

    :param input_graph:
    :param k: int
    :param replace: bool
    :param multi: bool
    :return:
    """

    while True:
        if multi:
            subgraph = edge_samp_multi(input_graph, k, replace)
        else:
            subgraph = edge_samp(input_graph, k, replace)

        selected_adj_mat, selected_users, selected_items = edge_list_to_matrix(subgraph)
        yield [selected_adj_mat, selected_users, selected_items]


"""
Pair sampling
"""


def pair_samp(graph, k):
    """
    sample a subgraph by choosing k pairs uniformly at random
    (differs from edge sampling because we allow ourselves to choose non-edges)
    :param graph: (nparray, nparray) of (edge_list, edge_weights)
    :param k: int
    :return: a list of pairs and weights, with weight=0 indicating a non-edge. WARNING: this is not the usual (edge list)
    graph structure
    """
    edge_list, weights = graph

    users = np.unique(edge_list[:, 0])
    selected_users = choice(users, k, replace=True)

    items = np.unique(edge_list[:, 1])
    selected_items = choice(items, k, replace=True)

    selected_pairs = np.c_[selected_users, selected_items]

    # populate sampled weights
    # this is very slow if done naively
    sort_by_user = edge_list[:, 0].argsort()
    el_sort = edge_list[sort_by_user]
    w_sort = weights[sort_by_user]

    selected_weights = np.zeros(k) # default to 0
    for i in range(k):

        user_start = np.searchsorted(el_sort[:,0], selected_pairs[i, 0])
        user_end = np.searchsorted(el_sort[:,0], selected_pairs[i, 0]+1)

        neighbours = el_sort[user_start:user_end,1]

        edge_selected = np.isin(neighbours, selected_pairs[i, 1])

        if edge_selected.any():
            selected_weights[i] = w_sort[user_start + np.where(edge_selected)]

    return (selected_pairs, selected_weights)


