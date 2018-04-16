"""
Code for network subsampling.
graphs are assumed to be represented as tuples (edge_lists, edge_weights)
"""

import numpy as np
from helpers.sampling_helpers import choice
from helpers.graph_helpers import isin_edgelist, user_neighbours_el, edge_list_to_matrix


def sample_nonedges(edge_list, n):
    """
    comparable speed to "sample_nonedges"
    :param edge_list:
    :param n:
    :return:
    """
    weights = np.ones(edge_list.shape[0], dtype=bool)
    a, users, items = edge_list_to_matrix((edge_list, weights))
    zero_ind_row, zero_ind_col = np.where(a==0)
    sel_zero_ind = choice(len(zero_ind_row), n, replace=True)
    zero_ind_users = users[zero_ind_row[sel_zero_ind]]
    zero_ind_items = items[zero_ind_col[sel_zero_ind]]
    return np.stack([zero_ind_users, zero_ind_items],1)


def sample_nonedges2(edge_list, n, users=None, items=None):
    """
    Takes the edge list of a graph and returns a random sample of n non-edges from the graph
    Returns an 'edge list' of the sampled non-edges

    :param graph:
    :return: adjacency matrix, users, items
    """

    ret_nel = np.zeros([n, 2], dtype=edge_list.dtype)  # holds returned edge list

    if users is None:
        users = np.unique(edge_list[:, 0])
    if items is None:
        items = np.unique(edge_list[:, 1])

    n_users = users.shape[0]
    n_items = items.shape[0]

    samp_u_idx = choice(n_users, n, replace=True)
    samp_i_idx = choice(n_items, n, replace=True)

    samp_el = np.zeros([n, 2], dtype=edge_list.dtype)
    samp_el[:, 0] = users[samp_u_idx]
    samp_el[:, 1] = items[samp_i_idx]

    # check which of the sampled pairs are actually non-edges
    is_zero = isin_edgelist(samp_el, edge_list, assume_unique=True, invert=True)
    num_zero = np.sum(is_zero)

    # collect the actual non-edges
    if not num_zero == 0:
        ret_nel[0:num_zero] = samp_el[is_zero]

    # sample replacements for any pairs that were actually edges
    if num_zero < n:
        ret_nel[num_zero:] = sample_nonedges(edge_list, n-num_zero, users, items)

    return ret_nel


def add_nonedges(generator, zd):
    """
    takes a generator that outputs graphs as edge lists + weights,
    and returns a generator that returns these samples, as well as (randomly sampled) non-edges

    Samples include a zd fraction of the non-edges

    :param input_graph:
    :param p:
    :param q:
    :param zd:
    :return:
    """
    genny = generator
    while True:
        el, w = next(genny)
        samp_users = np.unique(el[:,0])
        samp_items = np.unique(el[:,1])

        sampU = samp_users.shape[0]
        sampI = samp_items.shape[0]

        num_non_edges = np.random.binomial(sampU*sampI - el.shape[0], zd)
        # zero_el = sample_nonedges(el, num_non_edges, samp_users, samp_items)
        zero_el = sample_nonedges(el, num_non_edges)
        samp_el = np.concatenate([el, zero_el])
        samp_w = np.concatenate([w, np.zeros([num_non_edges,1], dtype=w.dtype)])

        yield (samp_el, np.squeeze(samp_w))


def add_masking_UI(generator, m_users, m_items):
    """
    takes a generator that outputs graphs as edge lists + (possibly 0) weights,
    and removes any pairs [i,j] where i in m_users and j in m_items

    intended use case: masking heldout data for training
    """
    genny = generator
    while True:
        el, w = next(genny)

        user_is_masked = np.isin(el[:, 0], m_users)
        item_is_masked = np.isin(el[:, 1], m_items)

        unmasked = np.invert(np.logical_and(user_is_masked, item_is_masked))

        yield (el[unmasked], w[unmasked])


def add_masking_el(generator, mask):
    """
    yields samples from the generator, deleting any edges that appear in 'mask'

    Note: this can result in a large slowdown for sample generation if
    1. the samples from the input generator are large, or
    2. a large fraction of the population users participate in at least 1 masked edge

    :param generator:
    :param mask:
    :return:
    """
    # dictionary w keys as users in the mask, and items their neighbours
    mask_neighbour_dict = user_neighbours_el(mask)
    m_users = np.fromiter(mask_neighbour_dict.keys(), dtype=np.int32)

    genny = generator
    while True:
        el, w = next(genny)

        user_sort = el[:, 0].argsort()
        el = el[user_sort]
        w = w[user_sort]

        users = np.unique(el[:, 0])

        users_in_mask = users[np.isin(users, m_users)]

        # for every user that appears in the mask edge list,
        # check if any of the edges it participates in are masked. If so, note the index in the edge list
        masked_ind = []
        for m_user in users_in_mask:
            start_index = np.searchsorted(el[:, 0], m_user)
            end_index = np.searchsorted(el[:, 0], m_user + 1)
            neighbours = el[start_index:end_index, 1]
            neighbours_masked = np.isin(neighbours, mask_neighbour_dict[m_user])
            masked_ind += [start_index + np.where(neighbours_masked)[0]]

        mi = np.concatenate(masked_ind)

        el_masked = np.delete(el, mi, axis=0)
        w_masked = np.delete(w, mi, axis=0)

        yield (el_masked, w_masked)


def add_masking_el2(generator, mask):
    """
    yields samples from the generator, deleting any edges that appear in 'mask'

    Note: this can result in a large slowdown for sample generation if
    1. the samples from the input generator are large, or
    2. a large fraction of the population users participate in at least 1 masked edge

    :param generator:
    :param mask:
    :return:
    """
    # dictionary w keys as users in the mask, and items their neighbours
    genny = generator
    while True:
        el, w = next(genny)

        unmasked_ind = isin_edgelist(el, mask, assume_unique=True, invert=True)

        el_masked = el[unmasked_ind]
        w_masked = w[unmasked_ind]

        yield (el_masked, w_masked)



def add_nonedges_and_masking_UI(generator, zd, m_users, m_items):
    genny = add_masking_UI(add_nonedges(generator, zd), m_users, m_items)
    while True:
        el, w = next(genny)
        yield el, w