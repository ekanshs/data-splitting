"""
Functions related to recommendation; mostly (entirely?) scoring metrics
Author: Victor Veitch
"""

import numpy as np


def nDCG(users, ranks, test, p=np.inf):
    """
    Computes the normalized Discounted Cumulative Gain at rank p

    Note: currently only handles simple graphs
    Note: if this turns out to be slow, can be sped up w smarter data structure and searchsorted in same way as pq-samp gen

    :param users: np.array of labels of user to score recommendations
    :param ranks: np.array of [users, recommendations]; ith row corresponds to user[i]
    :param test: edge list of true values to score against
    :param p: int
    :return:
    """

    # restrict rankings to top p
    rel_ranks = ranks[:, 0:min(p, ranks.shape[1])-1]

    nDCG = np.zeros(users.shape[0])
    for en, user in enumerate(users):
        user_test = test[test[:, 0] == user]
        test_ranks = np.isin(rel_ranks[en, :], user_test[:, 1]).nonzero()[0] + 1
        DCG = np.sum(np.log(2.) / (np.log(test_ranks + 1)))

        num_rel = min(p, user_test.shape[0])  # number of relevant
        itest_ranks = np.array(range(num_rel)) + 1
        iDCG = np.sum(np.log(2.) / (np.log(itest_ranks + 1)))
        if iDCG == 0:
            nDCG[en] = -13  # this is to avoid divide by 0 runtime erroe
        else:
            nDCG[en] = DCG / iDCG
        # if np.isnan(nDCG[en]):
        # happens if user has not rated any test items
        # (this is possible under pq sampling if all test ratings for the user are for items not included in train)
        #     pass

    return nDCG[nDCG != -13]


def precision_at_m(users, ranks, test, m):
    precision = np.zeros(users.shape[0])
    for en, user in enumerate(users):
        user_test = test[test[:, 0] == user]
        denominator = np.minimum(m, user_test[:, 1].shape[0])
        numerator = np.sum(np.isin(ranks[en, 0:m], user_test[:, 1])).astype(np.float)
        if denominator == 0:
            precision[en] = -13
        else:
            precision[en] = numerator/ denominator

    return precision[precision != -13]


def score_recommendations(rankings, test_users, test, p=1000, m=20, batch_size=500):
    """

    :param rankings: matrix of shape [test_users, num_rankings] where rankings[i,:] is ordered list of items in test
    :param test_users: test_users[i] is the label of the user represented in the ith row of the ranking matrix
    :param test: graph of test set, in (edge_list, weights) format
    :param p: as in NDCG@p
    :param m: as in precision@m
    :param batch_size: number of users to process per batch
    :return:
    """

    el, weights = test

    # batch up the users (to avoid memory problems... not clear how much this really matters)
    num_users = test_users.shape[0]
    num_splits = np.ceil(num_users / batch_size).astype(int)
    test_user_splits = np.array_split(test_users, num_splits)
    ranking_splits = np.array_split(rankings, num_splits)

    agg_ndcg_scores = []
    agg_precision_scores = []

    for split_idx in range(num_splits):
        test_users_sp = test_user_splits[split_idx]
        test_ranks_sp = ranking_splits[split_idx]

        agg_ndcg_scores += [nDCG(test_users_sp, test_ranks_sp, el, p=p)]
        agg_precision_scores += [precision_at_m(test_users_sp, rankings, el, m=m)]

    ndcg_scores = np.concatenate(agg_ndcg_scores)
    precision_scores = np.concatenate(agg_precision_scores)

    return ndcg_scores, precision_scores