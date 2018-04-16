"""
Author: Victor Veitch
"""

import numpy as np
import random


def choice(a, size=None, replace=False, p=None, seed=None):
    """
    Modified version of np.random.choice that greatly speeds up the (very important)
     case of sampling integers without replacement

     Warning: this relies on python3 range (would need to use xrange for python 2 version)
    :param a:
    :param size:
    :param replace:
    :param p:
    :param seed:
    :return:
    """

    if isinstance(a, np.integer):
        a_clean = np.asscalar(a)
    else:
        a_clean = a

    if isinstance(size, np.integer):
        size_clean = np.asscalar(size)
    else:
        size_clean = size

    if size is None:
        size_clean = 1

    if type(a_clean) == int \
            and type(size_clean) == int \
            and not replace \
            and p is None:
        if seed is not None:
            random.seed(seed)

        return np.array(random.sample(range(a_clean), size_clean))
    else:
        if seed is not None:
            np.random.seed(seed)

        return np.random.choice(a_clean, size_clean, replace, p)

