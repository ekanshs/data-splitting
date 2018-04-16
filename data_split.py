from __future__ import print_function

from graph_sampling import edge_samp

from graph_sampling import user_p_sample
from graph_sampling import item_p_sample
from helpers import row_major_order


def pq_samp_split(graph, p=0.8, q=0.8):
    graph_train, graph_holdout, tr_users = user_p_sample(graph, p, return_split=True)
    graph_lookup, graph_test, lu_items = item_p_sample(graph_holdout, q, return_split=True)

    # Arrange in row major order
    graph_train = row_major_order(graph_train)
    graph_lookup = row_major_order(graph_lookup)
    graph_test = row_major_order(graph_test)

    return graph_train, graph_lookup, graph_test, tr_users, lu_items


def edge_samp_split(graph, p=0.8):
    graph_train, graph_test = edge_samp(graph, int(p * len(graph[0])), return_split=True)

    # Arrange in row major order
    graph_train = row_major_order(graph_train)
    graph_test = row_major_order(graph_test)

    return graph_train, graph_test
