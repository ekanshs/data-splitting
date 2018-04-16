"""
Author: Victor Veitch
"""
import numpy as np


def softplus_inverse(y):
    return np.log(np.exp(y)-1.)


def isin1d_sorted(a, b):
    """
    equivalent to np.isin1d(a,b), except that it assumes b is sorted
    :param a:
    :param b:
    :return:
    """

    b_len = b.shape[0]
    search = np.searchsorted(b, a)
    return np.logical_not(search==b_len)


def row_major_order(graph):
    """
    :param edges:
    :return: indices of the graph in row major order
    """

    edges = graph[0].copy()
    row = edges[:, 0]
    col = edges[:, 1]
    indices = np.lexsort((col, row))

    return graph[0][indices], graph[1][indices]


def zero_index_sparse_graph(graph, axis=0, convert=None):
    if convert is None:
        zi_graph = graph[0].copy()
        convert = zero_index(zi_graph[:, axis])
        zi_graph[:, axis] = convert[zi_graph[:, axis]]
        return (zi_graph, graph[1]), convert
    else:
        # TODO: Check appropriate convert is passed in
        zi_graph = graph[0].copy()
        zi_graph[:, axis] = convert[zi_graph[:, axis]]
        return zi_graph, graph[1]


def zero_index(column, unique_column = False):
    """
    Zero index the given axis
    the mapping is label -> rank(label), e.g. [0, 7, 7, 4, 8] -> [0, 2, 2, 1, 3]
    Usage:
        >>> column = array([0,7,7,4,8])
        >>> zero_index(column)[column]

    Output: array([0,2,2,1,3])

    """
    if not unique_column:
        column = np.unique(column).copy()
    num_unique = column.shape[0]
    convert = np.zeros(np.max(column)+1, dtype=np.int32) - 1  # set default values to -1
    convert[column] = np.r_[0:num_unique]
    return convert

