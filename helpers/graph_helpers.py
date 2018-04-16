"""
Author: Victor Veitch
"""
import numpy as np
import tensorflow as tf

def matrix_to_edge_list(matrix):
    """
    takes a matrix as input, and returns a tuple (list of edges, edge_weights)
    e.g. edge_list[0] = [user_idx, item_idx, edge_val]
    :param matrix: nparray
    :return: nparray, edge list
    """

    edge_indices = matrix.nonzero()
    edge_weights = matrix[edge_indices]

    # edge_indices[k] should be index of kth edge
    edge_indices = np.vstack(edge_indices)
    edge_indices = edge_indices.T

    edge_weights = np.expand_dims(edge_weights, 1)

    return (edge_indices, edge_weights)


def user_item_flip(graph):
    """
    flips users and items
    useful sometimes for making code written to work on users easily work on items
    :param graph:
    :return:
    """

    edge_list, weights = graph

    el = np.copy(edge_list)
    el = el[:,[1,0]]

    return (el, np.copy(weights))


def edge_list_to_matrix(graph):
    """
    convert an edge list (of a sparse graph) into a minimal adjacency matrix
    where minimal means no rows or columns that are all 0

    Intended use case is p-sampled subgraph.

    Returns (adj_mat, users, items) where users and items are lists of the users and items

    The inverse operation is:
    edge_list_tmp, edge_weights = matrix_to_edge_list(adj_mat)
    user_list = users[edge_list_tmp[:,0]]
    item_list = items[edge_list_tmp[:,1]]
    edge_list = np.vstack((user_list,item_list)).T
    graph2 = (edge_list, edge_weights)

    (graph and graph2 are the same graph, but may have a different ordering of the edges)


    :param graph:
    :return: adjacency matrix, users, items
    """
    edge_list, weights = graph
    users = np.unique(edge_list[:, 0])
    items = np.unique(edge_list[:, 1])

    n_users = users.shape[0]
    n_items = items.shape[0]

    # relabel the users and items in the subgraph so that they're contiguous
    relabel = reindex_edge_list(edge_list, users, items)

    # the adjacency matrix
    adj_mat = np.zeros([n_users, n_items])
    u_index, i_index = relabel.T
    w = np.squeeze(np.copy(weights))
    adj_mat[(u_index, i_index)] = w

    return adj_mat, users, items


def reindex_edge_list(edge_list, users, items):
    """
    reindex the edge list so that users are relabelled as their index in users,
    and same for items

    Example: edge_list = [[4,2], [1,3], [12, 2]], users = [1, 4, 12], items = [2,3]
    then output is [[1, 0], [0, 1], [2, 0]]

    zero_indexing corresponds to reindex_edge_list(edge_list, np.unique(edge_list[:,0]), np.unique(edge_list[:,1]))

    Note that users[j] is the original label of the user labeled [j] in the reindexed graph
    :param edge_list:
    :param users:
    :param items:
    :return:
    """

    n_users = users.shape[0]
    n_items = items.shape[0]

    relabel = edge_list.copy()

    # the mapping is label -> rank(label), e.g. [0, 7, 7, 4, 8] -> [0, 2, 2, 1, 3]
    convert = np.zeros(np.max(users) + 1, dtype=np.int32)
    convert[users] = np.r_[0:n_users]
    relabel[:, 0] = convert[edge_list[:, 0]]

    # same for items
    convert = np.zeros(np.max(items) + 1, dtype=np.int32)
    convert[items] = np.r_[0:n_items]
    relabel[:, 1] = convert[edge_list[:, 1]]

    return relabel


def cartesian_product(*arrays):
    """
    cartesian product of the input arrays
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    # return arr
    return arr.reshape(-1, la)


def isin_edgelist2(edge_list, test_elements, assume_unique=False, invert=False):
    """
    check if test_elements are contained in edge list
    :param edge_list:
    :param test_elements:
    :param assume_unique: bool, passed on to the np.isin call
    :return:
    """
    assert edge_list.dtype == test_elements.dtype

    # transform to array of tuples
    el_view = np.ascontiguousarray(edge_list).view([('', edge_list.dtype)] * 2)
    te_view = np.ascontiguousarray(test_elements).view([('', test_elements.dtype)] * 2)

    return np.squeeze(np.isin(el_view, te_view, assume_unique=assume_unique, invert=invert))


def isin_edgelist(edge_list, test_elements, assume_unique=False, invert=False):
    """
    check if test_elements are contained in edge list
    :param edge_list:
    :param test_elements:
    :param assume_unique: bool, passed on to the np.isin call
    :return:
    """
    el_mod = edge_list[:,0] + 0.1*edge_list[:,1]
    te_mod = test_elements[:,0] + 0.1*test_elements[:,1]
    return np.isin(el_mod, te_mod, assume_unique=assume_unique, invert=invert)


def make_subarray_mask(mask, user_batch, item_batch):
    """
    Takes a mask for a large graph (as an edge list), and (subsampled) lists of users and items, and returns a binary
     array that masks the adjacency matrix of the user_batch x item_batch graph
    :param mask: nparray
    :param user_batch: nparray
    :param item_batch: nparray
    :return:
    """
    cp = cartesian_product(user_batch, item_batch) # all pairs in sample

    # get elements of the sample that are masked
    sample_mask = isin_edgelist(cp, mask)
    masked_elem = cp[sample_mask] # edge list of masked elements that exist in the subsample

    # index of masked elements in local adjacency matrix
    masked_elem_local = reindex_edge_list(masked_elem, user_batch, item_batch)
    mask_u, mask_i = masked_elem_local.T

    # make it an array
    U = user_batch.shape[0]
    I = item_batch.shape[0]
    sample_mask = np.zeros([U,I])
    sample_mask[mask_u, mask_i] = 1

    return sample_mask


def make_subarray_mask_closure(mask):
    """
    Takes a mask for a large graph (as an edge list), and (subsampled) lists of users and items, and returns a binary
     array that masks the adjacency matrix of the user_batch x item_batch graph
    :param mask: nparray
    :param user_batch: nparray
    :param item_batch: nparray
    :return
    """

    all_masked_users = np.unique(mask[:, 0])
    all_masked_items = np.unique(mask[:, 1])

    def make_subarray_mask(user_batch, item_batch):
        user_is_masked = np.isin(user_batch, all_masked_users, assume_unique=True)
        masked_users = user_batch[user_is_masked]
        item_is_masked = np.isin(item_batch, all_masked_items, assume_unique=True)
        masked_items = item_batch[item_is_masked]

        cp = cartesian_product(masked_users, masked_items) # all pairs in sample

        # fake an edge list w full contents to get a mask of the right shape
        unmasked_users = user_batch[np.logical_not(user_is_masked)]
        nuu = unmasked_users.shape[0]
        fake_user_edges = np.ones([nuu,2], dtype=np.int)
        fake_user_edges[:, 0] = unmasked_users
        fake_user_edges[:, 1] = item_batch[0]
        fake_user_weights = np.zeros(nuu)

        unmasked_items = item_batch[np.logical_not(item_is_masked)]
        nuu = unmasked_items.shape[0]
        fake_item_edges = np.ones([nuu, 2], dtype=np.int)
        fake_item_edges[:, 0] = user_batch[0]
        fake_item_edges[:, 1] = unmasked_items
        fake_item_weights = np.zeros(nuu)

        fake_el = np.concatenate([cp, fake_user_edges, fake_item_edges])
        fake_weights = np.concatenate([np.ones(cp.shape[0], dtype=np.int),
                                      fake_user_weights, fake_item_weights])

        return edge_list_to_matrix((fake_el, fake_weights))

    return make_subarray_mask


def comp_degs(graph):
    """
    Warning: assumes users and items are 0-indexed (i.e. users are labelled as 0 through max_user)
    :param graph:
    :return:
    """

    edge_list, weights = graph
    n_users = np.max(edge_list[:,0])+1
    n_items = np.max(edge_list[:,1])+1

    user_degs = np.zeros(n_users, dtype=weights.dtype)
    item_degs = np.zeros(n_items, dtype=weights.dtype)

    np.add.at(user_degs, edge_list[:,0], np.squeeze(weights))
    np.add.at(item_degs, edge_list[:,1], np.squeeze(weights))

    return user_degs, item_degs


def pick_users(graph, users, user_sorted=False):
    """

    For bipartite graphs

    :param graph:
    :param users:
    :return:
    """

    edge_list, weights = graph

    if user_sorted:
        el_sort = edge_list
        w_sort = weights
    else:
        # sort the edge list for fast indexing
        sort_by_user = edge_list[:, 0].argsort()
        el_sort = edge_list[sort_by_user]
        w_sort = weights[sort_by_user]

    user_start = np.searchsorted(el_sort[:,0], users)
    user_end = np.searchsorted(el_sort[:,0], users+1)

    selected_el = []
    selected_w = []
    for i in range(users.shape[0]):
        selected_el += [el_sort[user_start[i]:user_end[i]]]
        selected_w += [w_sort[user_start[i]:user_end[i]]]

    selected_el = np.concatenate(selected_el)
    selected_w = np.concatenate(selected_w)

    return (selected_el, selected_w)


def comp_degs_tf(graph, U, I):
    """
    Takes a weighted graph specified as an edge list and computes the weighted user and item degrees
    Actually somewhat broader: if edge_vals has shape [e, k] then we produce [u,k] and [i,k] values corresponding to the
     weighted degrees in each of the k graphs

    WARNING: this relies on undocumented behaviour of tf.scatter_nd, and may well break going forward (works as of tf1.3)

    :param weights: shape [edges, K]
    :param edge_list: shape [edges]
    :return user_degree, item_degree: shape [U,K] and [I,K] respectively
    """

    edge_list, weights = graph

    K = weights.shape.as_list()[1]

    # scatter_nd (as of version 1.1) adds duplicate indices
    user_degree = tf.scatter_nd(tf.expand_dims(edge_list[:, 0], axis=1), weights, [U, K])
    item_degree = tf.scatter_nd(tf.expand_dims(edge_list[:, 1], axis=1), weights, [I, K])

    return user_degree, item_degree


def neighbour_sum(edge_list, U, I, betas):
    beta_list = tf.gather(betas, edge_list[:, 1])
    sum_of_neighbours, _ = comp_degs_tf((edge_list, beta_list), U, I)
    return sum_of_neighbours


def user_init(graph, U, I, thetas):
    """
    Initializing the users independent of the graph structure leads to situations where the algorithm just learns
    to map every latent vector to the same value (the average rating). This is a hack to force users with common
    neighbours to have relatively similar values
    :param graph:
    :param U:
    :param I:
    :param thetas:
    :return:
    """

    edge_list, weights = graph

    thetas_list = tf.gather(thetas, edge_list[:, 0])
    _, item_neighbour_sums = comp_degs_tf((edge_list, thetas_list), U, I)

    theta_sum_list = tf.gather(item_neighbour_sums, edge_list[:, 1])
    new_theta_vals, _ = comp_degs_tf((edge_list, theta_sum_list), U, I)


def user_neighbours(graph):
    """
    returns neighbour_dictionary where neighbour_dictionary[i]=(neighbours of vertex i, edge weights connecting)
    :param graph:
    :return:
    """

    edge_list, weights = np.copy(graph[0]), np.copy(graph[1])

    # sort edge list by users
    user_sort = edge_list[:, 0].argsort()
    edge_list = edge_list[user_sort]
    weights = weights[user_sort]

    users = np.unique(edge_list[:, 0])
    neighbour_dict = {}

    last_index = 0
    for idx, user in enumerate(users):
        next_index = np.searchsorted(edge_list[:,0], user+1)
        neighbour_dict[user] = (edge_list[last_index:next_index, 1], weights[last_index:next_index])
        last_index = next_index

    return neighbour_dict


def user_neighbours_el(edge_list):
    """
    returns neighbour_dictionary where neighbour_dictionary[i]=(neighbours of vertex i)
    (same as neighbour users, except doesn't deal w edge weights)
    :param el: edge list
    :return:
    """

    # sort edge list by users
    user_sort = edge_list[:, 0].argsort()
    edge_list = edge_list[user_sort]

    users = np.unique(edge_list[:, 0])
    neighbour_dict = {}

    last_index = 0
    for user in users:
        next_index = np.searchsorted(edge_list[:,0], user+1)
        neighbour_dict[user] = np.sort(edge_list[last_index:next_index, 1])
        last_index = next_index

    return neighbour_dict


def item_neighbours(graph):
    return user_neighbours(user_item_flip(graph))