"""
Random walk sampler
(Hopefully just adapting code from node2vec; https://github.com/snap-stanford/snap/tree/master/examples/node2vec
"""

import numpy as np
from numpy.random import randint

from helpers.graph_helpers import user_neighbours, item_neighbours, edge_list_to_matrix
from helpers.sampling_helpers import choice


def sample_rw(graph, k):
    return next(rw_backtracking_el_generator(graph, k, 0.))


def rw_backtracking_el_generator(graph, k, p, q):
    """
    Return random walk of length 2k rooted at uniformly chosen user,
    in (edge_list, weights) form.
    Allows repeated edges.

    Random walk works as follows:
    At user:
    with probability q, return to most recent item
    else, select random neighbour
    At item:
    with probability p, return to most recent user.
    else, select random neighbour

    :param graph:
    :param num_roots:
    :param k:
    :return:
    """
    edge_list, weights = graph

    u_n = user_neighbours(graph)
    i_n = item_neighbours(graph)

    users = np.unique(edge_list[:, 0])

    while True:
        root = choice(users, 1)[0]

        # item that immediately preceded root in fictional walk, for easy backtracking
        candidate_prev_items, cpi_weights = u_n[root]
        prev_item_idx = np.random.randint(len(candidate_prev_items))
        prev_item = candidate_prev_items[prev_item_idx]
        edge_weight_from = cpi_weights[prev_item_idx]

        ret_el = np.zeros([2 * k, 2], dtype=np.int)
        ret_w = np.zeros(2 * k)

        cur_user = root
        for smp in range(0, 2 * k, 2):

            if np.random.binomial(1, q):
                # backtrack
                next_item = prev_item
                edge_weight_to = edge_weight_from
            else:
                candidate_items, ci_weights = u_n[cur_user]
                next_item_idx = np.random.randint(len(candidate_items))
                next_item = candidate_items[next_item_idx]
                edge_weight_to = ci_weights[next_item_idx]

            if np.random.binomial(1, p):
                # backtrack
                next_user = cur_user
                edge_weight_from = edge_weight_to
            else:
                candidate_users, cu_weights = i_n[next_item]
                next_user_idx = np.random.randint(len(candidate_users))
                next_user = candidate_users[next_user_idx]
                edge_weight_from = cu_weights[next_user_idx]

            ret_el[smp, 0] = cur_user
            ret_el[smp + 1, 0] = next_user
            ret_el[smp:smp + 2, 1] = next_item

            ret_w[smp] = edge_weight_to
            ret_w[smp + 1] = edge_weight_from

            cur_user = next_user
            prev_item = next_item


        yield (ret_el, ret_w)


def fast_rw_backtracking_el_generator(graph, k, p, q):
    """
    Return random walk of length 2k rooted at uniformly chosen user,
    in (edge_list, weights) form.
    Allows repeated edges.

    Random walk works as follows:
    At user:
    with probability q, return to most recent item
    else, select random neighbour
    At item:
    with probability p, return to most recent user.
    else, select random neighbour

    :param graph:
    :param num_roots:
    :param k:
    :return:
    """
    edge_list, weights = graph

    u_n = user_neighbours(graph)
    i_n = item_neighbours(graph)

    users = np.unique(edge_list[:, 0])

    while True:
        root = choice(users, 1)[0]

        ret_el = []
        ret_w = []

        cur_user = root
        for smp in range(0, 2 * k, 2):

            """
            Exploit that backtracking n times is equivalent to including n neighbours chosen wr,
            and continuing the walk from the last one
            """
            candidate_items, ci_weights = u_n[cur_user]
            u_degree = len(candidate_items)
            num_pick = 1 + np.random.negative_binomial(1, 1.-q)
            selected_idx = choice(u_degree, num_pick, replace=True)

            sel_edges = np.zeros([num_pick, 2], dtype=np.int)
            sel_edges[:, 0] = cur_user
            sel_edges[:, 1] = candidate_items[selected_idx]

            sel_weights = ci_weights[selected_idx]

            ret_el += [sel_edges]
            ret_w += [sel_weights]

            next_item = candidate_items[selected_idx][-1]

            """
            Same deal for picking users
            """
            candidate_users, cu_weights = i_n[next_item]
            i_degree = len(candidate_users)
            num_pick = 1 + np.random.negative_binomial(1, 1.-p)
            selected_idx = choice(i_degree, num_pick, replace=True)

            sel_edges = np.zeros([num_pick, 2], dtype=np.int)
            sel_edges[:, 0] = candidate_users[selected_idx]
            sel_edges[:, 1] = next_item

            sel_weights = cu_weights[selected_idx]

            ret_el += [sel_edges]
            ret_w += [sel_weights]

            next_user = candidate_users[selected_idx][-1]

            cur_user = next_user

        yield (np.concatenate(ret_el), np.concatenate(ret_w))


def rw_backtracking_generator(graph, k, p, q):
    """
    Return random walk of length 2k rooted at uniformly chosen user,
    in adjacency matrix form.

    Removes repeated edges

    Random walk works as follows:
    At user:
    select random neighbour
    At item:
    with probability p, return to most recent user.
    else, select random neighbour

    :param graph:
    :param num_roots:
    :param k:
    :return:
    """
    bt_el_gen = fast_rw_backtracking_el_generator(graph, k, p, q)

    while True:
        subgraph = next(bt_el_gen)
        selected_adj_mat, selected_users, selected_items = edge_list_to_matrix(subgraph)
        yield [selected_adj_mat, selected_users, selected_items]


def rw_w_completion_el_generator(graph, k):
    edge_list, weights = graph

    u_n = user_neighbours(graph)
    i_n = item_neighbours(graph)

    users = np.unique(edge_list[:, 0])

    while True:
        root = choice(users, 1)[0]

        sel_users = np.zeros(k+1, dtype=np.int)
        sel_items = np.zeros(k, dtype=np.int)

        sel_users[0] = root
        cur_user = root

        for smp in range(k):
            candidate_items, _ = u_n[cur_user]
            next_item = candidate_items[randint(candidate_items.shape[0])]

            candidate_users,_ = i_n[next_item]
            next_user = candidate_users[randint(candidate_users.shape[0])]

            sel_users[smp+1] = next_user
            sel_items[smp] = next_item

            cur_user = next_user

        ret_el = []
        ret_w = []

        selected_items = np.unique(sel_items)

        for user in sel_users:
            neighbours, neighbours_weights = u_n[user]
            incl_neighbours_bl = np.isin(neighbours, selected_items, assume_unique=True)

            inc_neigh = neighbours[incl_neighbours_bl]
            inc_weights = neighbours_weights[incl_neighbours_bl]

            inc_el = np.zeros([inc_neigh.shape[0], 2], dtype=np.int)
            inc_el[:, 0] = user
            inc_el[:, 1] = inc_neigh

            ret_el += [inc_el]
            ret_w += [inc_weights]

        yield np.concatenate(ret_el), np.concatenate(ret_w)


def rw_w_completion_generator(graph, k):
    rwc_el_gen = rw_w_completion_el_generator(graph, k)

    while True:
        subgraph = next(rwc_el_gen)
        selected_adj_mat, selected_users, selected_items = edge_list_to_matrix(subgraph)
        yield [selected_adj_mat, selected_users, selected_items]
