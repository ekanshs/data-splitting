"""
Author: Victor Veitch
"""
import os

import numpy as np
import pandas as pd


# ML_DIR = '/home/victor/Documents/network_sampling/weird_sgd/data/ml-20m'
#
# m_cols = ['movie_id', 'title', 'genres']
# movies = pd.read_csv(os.path.join(ML_DIR, 'movies.csv'), sep=',', names=m_cols, encoding='latin-1', header=0)
# movie_titles = movies.as_matrix(['movie_id', 'title'])
# ML_titles_dict = dict(zip(movie_titles[:,0], movie_titles[:,1]))

def print_top_items(VT, item_names_dict, orig_i_indexing, n_top_items):
    for topic_idx, topic in enumerate(VT):
        message = "Topic #{}: ".format(topic_idx)
        top_items = np.argsort(topic)[:-n_top_items - 1:-1]
        top_item_orig_ind = orig_i_indexing[top_items]
        top_item_names = [item_names_dict[itm_idx] for itm_idx in top_item_orig_ind]

        message += " ".join(top_item_names)
        print(message)
    print()


def items_to_names(items, item_names_dict, orig_i_indexing):
    orig_idx = orig_i_indexing[items]
    item_names = [item_names_dict[itm_idx] for itm_idx in orig_idx]
    return item_names