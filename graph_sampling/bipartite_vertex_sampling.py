"""
Vertex sampling
(including degree biased sampling)
"""

import numpy as np
import scipy.sparse as sparse

from helpers.graph_helpers import reindex_edge_list
from helpers.sampling_helpers import choice


def vert_samp(graph, k, l, u_dist = None, i_dist = None):
    """
    Sampling defaults to with replacement because multinomial sampling is easier that multidimensional hypergeometric

    :param graph: graph in usual (edge_list , weights) format
    :param k: number of users in sample
    :param l: number of items in sample
    :param u_dist: sampling distribution for users (defaults to uniform with replacement)
    :param i_dist: sampling distribution for items (defaults to uniform with replacement)
    :return: adjacency matrix of the subsample, and lists of selected users and selected items
    """
    edge_list, weights = graph

    users = np.unique(edge_list[:, 0])
    items = np.unique(edge_list[:,1])

    if u_dist is None:
        selected_users = choice(users, k, replace=True)
    else:
        proto_selected_users = np.random.multinomial(k, u_dist)
        selected_users_list = []
        # build a list of selected users, where users that are selected multiple times are repeated as required
        for mult in range(proto_selected_users.max()):
            selected_users_list += np.where(proto_selected_users > mult)
        selected_users = np.concatenate(selected_users_list)

    if i_dist is None:
        selected_items = choice(items, l, replace=True)
    else:
        proto_selected_items = np.random.multinomial(l, i_dist)
        selected_items_list = []
        # build a list of selected users, where users that are selected multiple times are repeated as required
        for mult in range(proto_selected_items.max()):
            selected_items_list += np.where(proto_selected_items > mult)
        selected_items = np.concatenate(selected_items_list)

    # for with replacement sampling
    selected_users, u_cts = np.unique(selected_users, return_counts=True)
    selected_items, i_cts = np.unique(selected_items, return_counts=True)

    # get the non-zero entries of the subsample
    u_edge_selected = np.in1d(edge_list[:,0], selected_users)
    samp_el = np.copy(edge_list[u_edge_selected])
    i_edge_selected = np.in1d(samp_el[:,1], selected_items)
    samp_el = samp_el[i_edge_selected]

    samp_w = np.copy(weights[u_edge_selected])
    samp_w = samp_w[i_edge_selected]

    # construct the corresponding adjacency matrix (which can have all 0 rows or columns)

    # contiguous relabelling of the edge list
    relabel_u, relabel_i = reindex_edge_list(samp_el, selected_users, selected_items).T

    # for each all 0 user j, add phantom edge [j,0] of weight 0; this hack allows for all zero rows and columns
    zero_users = np.isin(range(selected_users.size), relabel_u, invert=True).nonzero()[0] # selected users w no edges
    relabel_u = np.append(relabel_u, zero_users)
    relabel_i = np.append(relabel_i, np.zeros_like(zero_users))
    samp_w = np.append(samp_w, np.zeros_like(zero_users, dtype=np.float32))

    # and same for items
    zero_items = np.isin(range(selected_items.size), relabel_i, invert=True).nonzero()[0] # selected items w no edges
    relabel_i = np.append(relabel_i, zero_items)
    relabel_u = np.append(relabel_u, np.zeros_like(zero_items))
    samp_w = np.append(samp_w, np.zeros_like(zero_items, dtype=np.float32))

    # add in the required copies of the users that were selected multiple times
    dup_ru, dup_ri, dup_w = _add_mult_samp_users(selected_users, k, u_cts, relabel_u, relabel_i, weights)

    relabel_u = np.append(relabel_u, dup_ru)
    relabel_i = np.append(relabel_i, dup_ri)
    samp_w = np.append(samp_w, dup_w)

    # add in the required copies of items that were selected multiple times (due to with replacement sampling)
    dup_ri, dup_ru, dup_w = _add_mult_samp_users(selected_items, l, i_cts, relabel_i, relabel_u, weights)

    relabel_u = np.append(relabel_u, dup_ru)
    relabel_i = np.append(relabel_i, dup_ri)
    samp_w = np.append(samp_w, dup_w)

    adj_mat = np.zeros([k,l])
    adj_mat[relabel_u, relabel_i] = np.squeeze(samp_w)

    return adj_mat, selected_users, selected_items


def _add_mult_samp_users(selected_users, k, u_cts, relabel_u, relabel_i, weights):
    """
    helper function for vertex sampling with replacement
    computes extra edges and users required to account for the with replacement sampling

    :param selected_users:
    :param k:
    :param u_cts:
    :param relabel_u:
    :param relabel_i:
    :param weights:
    :return:
    """

    # add in the required copies of users that were selected multiple times (due to with replacement sampling)
    n_u = selected_users.shape[0]
    full_selected_users = np.append(selected_users, np.repeat(-1, k - n_u))
    dup_ru = []
    dup_ri = []
    dup_w = []
    for dup in (u_cts > 1).nonzero()[0]:
        dup_inc = np.isin(relabel_u, dup).nonzero()[0]
        for _ in range(u_cts[dup]-1):
            full_selected_users[n_u] = dup

            # Note: every selected user must have at least one edge in the subgraph
            dup_ru += [np.repeat(n_u, dup_inc.shape[0])] # new user w label n_u
            # clone the edges for the new user
            dup_ri += [np.copy(relabel_i[dup_inc])]
            dup_w += [np.copy(weights[dup_inc])]
            n_u += 1

    return np.concatenate(dup_ru), np.concatenate(dup_ri), np.concatenate(dup_w)


def vert_samp_generator(graph, k, l, u_dist = None, i_dist = None):
    """

    :param graph:
    :param k:
    :param l:
    :param u_dist:
    :param i_dist:
    :return:
    """
    while True:
        yield vert_samp(graph, k, l, u_dist, i_dist)


def fast_vert_samp_generator(graph, k, l, u_dist = None, i_dist = None):
    """
    :param graph:
    :param k:
    :param l:
    :param u_dist:
    :param i_dist:
    :return:
    """

    edge_list, weights = np.copy(graph[0]), np.copy(graph[1])

    sparse_rep = sparse.coo_matrix((np.squeeze(weights), (edge_list[:,0], edge_list[:,1]))).tocsr()

    users = np.unique(edge_list[:,0])
    items = np.unique(edge_list[:,1])

    while True:
        if u_dist is None:
            selected_users = choice(users, k, replace=True)
        else:
            proto_selected_users = np.random.multinomial(k, u_dist)
            selected_users_list = []
            # build a list of selected users, where users that are selected multiple times are repeated as required
            for mult in range(proto_selected_users.max()):
                selected_users_list += np.where(proto_selected_users > mult)
            selected_users = np.concatenate(selected_users_list)

        if i_dist is None:
            selected_items = choice(items, l, replace=True)
        else:
            proto_selected_items = np.random.multinomial(l, i_dist)
            selected_items_list = []
            # build a list of selected users, where users that are selected multiple times are repeated as required
            for mult in range(proto_selected_items.max()):
                selected_items_list += np.where(proto_selected_items > mult)
            selected_items = np.concatenate(selected_items_list)

        samp = sparse_rep[selected_users]
        samp = samp[:, selected_items]

        yield samp.toarray(), selected_users, selected_items
