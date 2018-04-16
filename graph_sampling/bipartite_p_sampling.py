import numpy as np

from helpers.graph_helpers import user_item_flip, edge_list_to_matrix
from helpers.sampling_helpers import choice

"""
p-sampling

graphs are assumed to be represented as tuples (edge_lists, edge_weights)
"""


def user_p_sample(graph, p, users = None, return_split = False, seed = None):
    """
    :param graph: (nparray, nparray) of (edge_list, edge_weights)
    :param p: 0 < p < 1
    :param return_split: bool indicates whether to also return the complement of the sample, false by default
    :param seed: int
    :return: partition of the edge list into [user-samp(graph,p), G\\user-samp(graph,p)]
    """

    edge_list, weights = graph

    if users is None:
        users = np.unique(edge_list[:, 0])

    U = users.shape[0]
    if seed is not None:
        np.random.seed(seed)
    pick_number = np.random.binomial(U, p)
    selected_users = choice(users, pick_number, replace=False, seed=seed)

    in_selection = np.in1d(edge_list[:,0], selected_users)

    if not return_split:
        return (np.copy(edge_list[in_selection]), np.copy(weights[in_selection]))
    else:
        out_selection = np.invert(in_selection)
        return (np.copy(edge_list[in_selection]), np.copy(weights[in_selection])), \
            (np.copy(edge_list[out_selection]), np.copy(weights[out_selection])), \
            selected_users


def user_p_sample2(graph, p, return_split = False, seed = None):
    """
    Note: this is actually only slightly (~40%) faster than naive sampling

    :param graph: (nparray, nparray) of (edge_list, edge_weights)
    :param p: 0 < p < 1
    :param return_split: bool indicates whether to also return the complement of the sample, false by default
    :param seed: int
    :return: partition of the edge list into [user-samp(graph,p), G\\user-samp(graph,p)]
    """

    edge_list, weights = np.copy(graph[0]), np.copy(graph[1])

    # sort edge list by users
    user_sort = edge_list[:, 0].argsort()
    edge_list = edge_list[user_sort]
    weights = weights[user_sort]

    users = np.unique(edge_list[:, 0])
    U = users.shape[0]

    # edge_list[user_indices[j][0], user_indices[j][1]]  is edge list of all edges that include user j
    user_indices = np.zeros([U, 2], dtype=np.int32)
    user_first_occ = np.searchsorted(edge_list[:, 0], users+1)
    user_indices[1:, 0] = user_first_occ[:-1]
    user_indices[:,1] = user_first_occ

    # equivalent, but easier to understand:
    # last_index = 0
    # user_indices = np.zeros([U, 2], dtype=np.int32)
    # for idx, user in enumerate(users):
    #     next_index = np.searchsorted(edge_list[:, 0], user+1)
    #     user_indices[idx] = last_index, next_index
    #     last_index = next_index

    if seed is not None:
        np.random.seed(seed)

    # select the users
    pick_number = np.random.binomial(U, p)
    selected_users_indices = choice(U, pick_number, replace=False)

    # construction of the sample of edges that connect
    index_pair_list = user_indices[selected_users_indices].tolist()
    sample_edge_list = np.concatenate([edge_list[start:finish] for start, finish in index_pair_list])
    sample_weights = np.concatenate([weights[start:finish] for start, finish in index_pair_list])
    sample = (sample_edge_list, sample_weights)

    if not return_split:
        return sample
    else:
        remaining_user_indices = np.where(np.in1d(np.arange(U), selected_users_indices, invert=True))[0]
        index_pair_list = user_indices[remaining_user_indices].tolist()
        rem_sample_edge_list = np.concatenate([edge_list[start:finish] for start, finish in index_pair_list])
        rem_sample_weights = np.concatenate([weights[start:finish] for start, finish in index_pair_list])
        rem_sample = (rem_sample_edge_list, rem_sample_weights)

        return sample, rem_sample


def item_p_sample(graph, p, items = None, return_split=False, seed=None):
    """
    :param graph: (nparray, nparray) of (edge_list, edge_weights)
    :param p: 0 < p < 1
    :return: partition of the edge list into [item-samp(graph,q), G\\user-samp(graph,q)]
    """

    if not return_split:
        # flip and pass to user sampler
        selected = user_p_sample(user_item_flip(graph), p, users=items, return_split=False, seed=seed)

        # unflip before return
        return user_item_flip(selected)
    else:
        selected, not_selected, selected_items = user_p_sample(user_item_flip(graph), p, users=items, return_split=True, seed=seed)

        return user_item_flip(selected), user_item_flip(not_selected), selected_items


def cond_user_p_sample(graph, p, cond_user, seed = None):
    """

    :param graph: (nparray, nparray) of (edge_list, edge_weights)
    :param p: 0 < p < 1
    :param cond_user: user to include
    :param seed: int
    :return: partition of the edge list into [user-samp(graph,p), G\\user-samp(graph,p)]
    """

    edge_list, weights = np.copy(graph[0]), np.copy(graph[1])

    # sort edge list by users
    user_sort = edge_list[:, 0].argsort()
    edge_list = edge_list[user_sort]
    weights = weights[user_sort]

    users = np.unique(edge_list[:, 0])
    U = users.shape[0]

    # edge_list[user_indices[j][0], user_indices[j][1]]  is edge list of all edges that include user j
    user_indices = np.zeros([U, 2], dtype=np.int32)
    user_first_occ = np.searchsorted(edge_list[:, 0], users+1)
    user_indices[1:, 0] = user_first_occ[:-1]
    user_indices[:,1] = user_first_occ

    if seed is not None:
        np.random.seed(seed)

    # select the additional users
    pick_number = np.random.binomial(U, p)
    selected_users_indices = choice(U, pick_number, replace=False)

    # add in the user we're conditioning on selecting
    cond_user_idx = np.searchsorted(users, cond_user)
    if cond_user_idx not in selected_users_indices:
        selected_users_indices[0] = cond_user_idx

    # construction of the sample of edges that connect
    index_pair_list = user_indices[selected_users_indices].tolist()
    sample_edge_list = np.concatenate([edge_list[start:finish] for start, finish in index_pair_list])
    sample_weights = np.concatenate([weights[start:finish] for start, finish in index_pair_list])
    sample = (sample_edge_list, sample_weights)

    return sample


def cond_item_p_sample(graph, p, cond_item, seed=None):
    """
    p-samples items, conditional on cond_item being included

    :param graph: (nparray, nparray) of (edge_list, edge_weights)
    :param p: 0 < p < 1
    :param cond_item: condition on this item being included in sample
    :return conditional item p-ssample
    """

    # flip and pass to user sampler
    selected = cond_user_p_sample(user_item_flip(graph), p, cond_user=cond_item, seed=seed)

    # unflip before return
    return user_item_flip(selected)


def pq_sample(graph, p, q, users=None, items=None, seed=None):
    """
    returns a random p,q-sampled subgraph of graph
    :param graph:
    :param p:
    :param q:
    :return:
    """
    return item_p_sample(user_p_sample(graph, p, users=users, return_split=False, seed=seed), q,
                         items=items, return_split=False, seed=seed)


def pq_sample_generator(input_graph, p, q):
    """
    generates pq subsamples from input_graph, returned as adjacency matrices.
    Each sample also returns arrays giving the identities of the selected vertices and items in input_graph

    :param input_graph:
    :param p:
    :param q:
    :return:
    """

    while True:
        subgraph = pq_sample(input_graph, p, q)
        selected_adj_mat, selected_users, selected_items = edge_list_to_matrix(subgraph)
        yield [selected_adj_mat, selected_users, selected_items]


def fast_pq_sample_generator(input_graph, p, q):
    """
    generates pq subsamples from input_graph, returned as adjacency matrices.

    Each sample also returns arrays giving the identities of the selected vertices and items in input_graph

    :param input_graph:
    :param p:
    :param q:
    :return:
    """

    edge_list, weights = np.copy(input_graph[0]), np.copy(input_graph[1])

    # change the representation of the graph to allow for faster sampling

    # sort edge list by items
    item_sort = edge_list[:, 1].argsort()
    edge_list = edge_list[item_sort]
    weights = weights[item_sort]

    items = np.unique(edge_list[:, 1])
    I = items.shape[0]

    # edge_list[item_indices[j][0], item_indices[j][1]]  is edge list of all edges that include item j
    last_index = 0
    item_indices = np.zeros([I,2], dtype=np.int32)
    for idx, item in enumerate(items):
        next_index = np.searchsorted(edge_list[:,1], item+1)
        item_indices[idx] = last_index, next_index
        last_index = next_index

    while True:
        # q-sample using efficient representation
        pick_number = np.random.binomial(I, q)
        selected_item_indices = choice(I, pick_number, replace=False)
        index_pair_list = item_indices[selected_item_indices].tolist()
        q_sample_edge_list = np.concatenate([edge_list[start:finish] for start,finish in index_pair_list])
        q_sample_weights = np.concatenate([weights[start:finish] for start, finish in index_pair_list])
        q_sample = (q_sample_edge_list, q_sample_weights)

        # p-sample naively (but now on *much* smaller graph)
        subgraph = user_p_sample2(q_sample, p)

        yield subgraph


def fast_pq_sample_generator_adjmat(input_graph, p, q):
    genny = fast_pq_sweep_generator(input_graph, p, q)
    while True:
        subgraph = next(genny)
        selected_adj_mat, selected_users, selected_items = edge_list_to_matrix(subgraph)
        yield [selected_adj_mat, selected_users, selected_items]


def user_sweep_generator(input_graph, p):
    """
    randomly partitions users into k=ceil(1/p) bins, and iterates through the subgraphs induced by restricting to
    the first k-1 of these
    :param input_graph:
    :param p:
    :return:
    """
    edge_list, weights = np.copy(input_graph[0]), np.copy(input_graph[1])

    # change the representation of the graph to allow for faster sampling

    # sort edge list by users
    user_sort = edge_list[:, 0].argsort()
    edge_list = edge_list[user_sort]
    weights = weights[user_sort]

    users = np.unique(edge_list[:, 0])
    U = users.shape[0]

    # edge_list[user_indices[j][0], user_indices[j][1]]  is edge list of all edges that include user j
    last_index = 0
    user_indices = np.zeros([U, 2], dtype=np.int32)
    for idx, user in enumerate(users):
        next_index = np.searchsorted(edge_list[:,0], user+1)
        user_indices[idx] = last_index, next_index
        last_index = next_index

    # partition the user indices to sweep through
    ps = np.repeat(p, np.floor(1./p))
    ps = np.concatenate([ps, np.array([1. - np.sum(ps)])])
    num_in_part = np.random.multinomial(U, ps)

    splits = np.array_split(np.random.permutation(U), np.cumsum(num_in_part))

    for selected_users in splits[:-2]:  # -2 since last is empty, and second last is the 'leftovers' partition
        index_pair_list = user_indices[selected_users].tolist()
        p_sample_edge_list = np.concatenate([edge_list[start:finish] for start,finish in index_pair_list])
        p_sample_weights = np.concatenate([weights[start:finish] for start, finish in index_pair_list])
        p_sample = (p_sample_edge_list, p_sample_weights)
        yield p_sample


def fast_pq_sweep_generator(input_graph, p, q):
    """
    sweep through p,q-subgraphs of input_graph, returning each subgraph as adjacency matrix
    pseudo code:
    1. randomly partition items
    2. for each item_partition:
        1. randomly partition users
        2. for each user_parititon
            return graph[user_partition, item_partition]

    That is, we pick a subset of the items, sweep through all the users with those items held fixed, and repeat

    Each sample also returns arrays giving the identities of the selected vertices and items in input_graph

    :param input_graph:
    :param p:
    :param q:
    :return:
    """

    edge_list, weights = np.copy(input_graph[0]), np.copy(input_graph[1])

    # change the representation of the graph to allow for faster sampling

    # sort edge list by items
    item_sort = edge_list[:, 1].argsort()
    edge_list = edge_list[item_sort]
    weights = weights[item_sort]

    items = np.unique(edge_list[:, 1])
    I = items.shape[0]

    # edge_list[item_indices[j][0], item_indices[j][1]]  is edge list of all edges that include item j
    last_index = 0
    item_indices = np.zeros([I,2], dtype=np.int32)
    for idx, item in enumerate(items):
        next_index = np.searchsorted(edge_list[:,1], item+1)
        item_indices[idx] = last_index, next_index
        last_index = next_index

    while True:
        # partition the item indices to sweep through
        qs = np.repeat(q, np.floor(1. / q))
        qs = np.concatenate([qs, np.array([1. - np.sum(qs)])])
        num_in_part = np.random.multinomial(I, qs)

        splits = np.array_split(np.random.permutation(I), np.cumsum(num_in_part))

        for selected_items in splits[:-2]:  # -2 since last is empty, and second last is the 'leftovers' partition
            index_pair_list = item_indices[selected_items].tolist()
            q_sample_edge_list = np.concatenate([edge_list[start:finish] for start, finish in index_pair_list])
            q_sample_weights = np.concatenate([weights[start:finish] for start, finish in index_pair_list])
            q_sample = (q_sample_edge_list, q_sample_weights)

            # now sweep through users
            user_sweep = user_sweep_generator(q_sample, p)
            for subgraph in user_sweep:
                yield subgraph


def fast_pq_sweep_generator_adjmat(input_graph, p, q):
    genny = fast_pq_sweep_generator(input_graph, p, q)
    while True:
        subgraph = next(genny)
        selected_adj_mat, selected_users, selected_items = edge_list_to_matrix(subgraph)
        yield [selected_adj_mat, selected_users, selected_items]


def cond_pq_sample_generator_closure(input_graph, p, q):
    """
    returns a generator that takes a user as input, and generates p,q subsamples conditional on that user being
    included in the graph

    :param input_graph:
    :param p:
    :param q:
    :return:
    """

    edge_list, weights = np.copy(input_graph[0]), np.copy(input_graph[1])

    # change the representation of the graph to allow for faster sampling

    # sort edge list by users
    user_sort = edge_list[:, 0].argsort()
    edge_list = edge_list[user_sort]
    weights = weights[user_sort]

    users = np.unique(edge_list[:, 0])
    U = users.shape[0]

    # edge_list[user_indices[j][0], user_indices[j][1]]  is edge list of all edges that include user j
    last_index = 0
    user_indices = np.zeros([U,2], dtype=np.int32)
    for idx, user in enumerate(users):
        next_index = np.searchsorted(edge_list[:,0], user+1)
        user_indices[idx] = last_index, next_index
        last_index = next_index

    def conditional_pq_gen(cond_user):
        # p,q sample conditioned on cond_user being included
        cond_user_idx = np.searchsorted(users, cond_user)
        while True:
            # p-sample using efficient representation
            pick_number = np.random.binomial(U, p)
            selected_user_indices = choice(U, pick_number, replace=False)

            # add in the user we're conditioning on selecting
            cond_user_idx = np.searchsorted(users, cond_user)
            if cond_user_idx not in selected_user_indices:
                selected_user_indices[0] = cond_user_idx

            index_pair_list = user_indices[selected_user_indices].tolist()
            p_sample_edge_list = np.concatenate([edge_list[start:finish] for start,finish in index_pair_list])
            p_sample_weights = np.concatenate([weights[start:finish] for start, finish in index_pair_list])
            p_sample = (p_sample_edge_list, p_sample_weights)

            # to force the user to be in the graph we must also ensure at least one of its neighbours is included
            cond_user_neighbours = edge_list[user_indices[cond_user_idx][0]:user_indices[cond_user_idx][1]][:,1]
            cond_neighbour = choice(cond_user_neighbours)

            # q-sample naively (but now on *much* smaller graph)
            subgraph = cond_item_p_sample(p_sample, q, cond_neighbour)
            selected_adj_mat, selected_users, selected_user_indices = edge_list_to_matrix(subgraph)
            yield [selected_adj_mat, selected_users, selected_user_indices]

    return conditional_pq_gen


def clean_item_p_sample(graph, p):
    """
    p sample the items to create a train-test split, and then clean up the resulting test set so that
    test contains only users that are also in train
    Note that we *do not* zero index the items (because these will be passed in to something that contains the full item set)
    :param graph:
    :param p: train set is p-sampling of items of graph
    :return:
    """
    lazy = np.copy(graph)
    # interchange users and items
    lazy[:,0] = graph[:,1]
    lazy[:,1] = graph[:,0]

    ltrain, ltest = user_p_sample(lazy, p)
    # eliminate any users in test that aren't also in train, and then give those users a new common zero index
    # do not reindex the items! (that would break prediction code)
    ltrain, ltest = clean_p_samp_split(ltrain, ltest, zi_train_u=False, zi_test_u=False)

    train = ltrain.copy()
    train[:,0] = ltrain[:,1]
    train[:,1] = ltrain[:,0]

    test = ltest.copy()
    test[:,0] = ltest[:,1]
    test[:,1] = ltest[:,0]

    return train, test


def clean_p_samp_split(train, test, zi_items=True, zi_train_u=True, zi_test_u=True):
    """
    Zero index + remove test edges with movies that are not in train

    Return a version of test that contains only items that are also in train
    (to avoid trying to make predictions on unknown items)
    and also relabel the users and items st the sets are contiguous (to avoid degree 0 vertices)

    :param train: nparray of (multi-)edges, with train[0] = [user_idx, item_idx, edge_val]
    :param test: nparray of (multi-)edges, with test[0] = [user_idx, item_idx, edge_val]
    :return: cleaned test and train,
    and arrays giving the mapping of user and item indices in the cleaned set to indices in the set passed in
    """

    # Zero index Users in Train
    if zi_train_u:
        uids_train = train[:,0]
        allusers_train = np.unique(uids_train)
        numusers_train = allusers_train.size
        convertusers_train = np.zeros(np.max(allusers_train)+1, dtype=np.int64)
        convertusers_train[allusers_train] = np.r_[0:numusers_train]
        train[:,0] = convertusers_train[uids_train]

    # Zero index Users in Test
    if zi_test_u:
        uids_test = test[:,0]
        allusers_test = np.unique(uids_test)
        numusers_test = allusers_test.size
        convertusers_test = np.zeros(np.max(allusers_test)+1, dtype=np.int64)
        convertusers_test[allusers_test] = np.r_[0:numusers_test]
        test[:,0] = convertusers_test[uids_test]

    # Remove edges in test if item is not in train
    train_items = np.unique(train[:,1])
    test = test[np.in1d(test[:,1], train_items)]

    # Zero index items (iids: item ids)
    if zi_items:
        iids = train[:,1]
        allitems = np.unique(train[:,1])
        numitems = allitems.size
        convertitems = np.zeros(np.max(allitems)+1, dtype=np.int64)
        convertitems[allitems] = np.r_[0:numitems]
        train[:,1] = convertitems[iids]
        test[:,1] = convertitems[test[:,1]]

    if zi_items and zi_train_u:
        # return allusers_train and allitems to allow us to track the ground truth params corresponding to the training set
        return train, test, allusers_train, allitems
    else:
        return train, test
