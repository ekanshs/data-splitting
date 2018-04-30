import numpy as np
from sklearn.decomposition import NMF
from trainedNMF import trainedNMF
from scipy.sparse import csr_matrix
import pickle
import torch

from helpers.misc import zero_index, zero_index_sparse_graph

from data_split import pq_samp_split

from helpers import nDCG, precision_at_m

# Load data
loc = "/Users/ekansh/repos/data/{}"
ds = "nyt/"
print("Dataset is : {}".format(ds))
loss = 'kullback-leibler'
# loss = "frobenius"
#
with open(loc.format(ds) + 'data.pkl') as f:
    data = pickle.load(f)

graph = (data[:, :2], data[:, 2])

graph, _ = zero_index_sparse_graph(graph, axis=0)
graph, _ = zero_index_sparse_graph(graph, axis=1)

# Data Split
tr_graph, lu_graph, ts_graph, tr_U, lu_I = pq_samp_split(graph)
U = np.unique(graph[0][:, 0])
nU = U.shape[0]

# THIS IS CONFUSING. FIX IT!
tr_U_zero_indexer = zero_index(tr_U, True)
I = np.unique(graph[0][:, 1])
nI = I.shape[0]
n_tr_U = tr_U.shape[0]
tr_I = I
n_tr_I = nI
lu_U = ts_U = np.setdiff1d(U, tr_U, assume_unique=True)
n_lu_U = n_ts_U = lu_U.shape[0]
lu_U_zero_indexer = ts_U_zero_indexer = zero_index(lu_U, True)
n_lu_I = lu_I.shape[0]
lu_I_zero_indexer = zero_index(lu_I, True)
ts_I = np.setdiff1d(I, lu_I, assume_unique=True)
n_ts_I = ts_I.shape[0]
ts_I_zero_indexer = zero_index(ts_I, True)

## Training begins
K = 10

# NMF of train set:
# Zero index train set
zi_tr_graph = zero_index_sparse_graph(tr_graph, axis=0, convert=tr_U_zero_indexer)
zi_tr_graph_sparse = csr_matrix((zi_tr_graph[1], (zi_tr_graph[0][:, 0], zi_tr_graph[0][:, 1])), shape=(n_tr_U, n_tr_I))

print("Using {} loss".format(loss))

tr_model = NMF(n_components=K, init='random', random_state=0, max_iter=1000, beta_loss=loss, solver='mu')
tr_U_f = tr_model.fit_transform(zi_tr_graph_sparse)  # user features
tr_I_f = tr_model.components_  # item features

# Train lookup model with item features fixed
# 2. Train on graph_lookup: fix item_feat
zi_lu_graph = zero_index_sparse_graph(lu_graph, axis=1, convert=lu_I_zero_indexer)
zi_lu_graph = zero_index_sparse_graph(zi_lu_graph, axis=0, convert=lu_U_zero_indexer)
zi_lu_graph_sparse = csr_matrix((zi_lu_graph[1], (zi_lu_graph[0][:, 0], zi_lu_graph[0][:, 1])), shape=(n_lu_U, n_lu_I))

lu_model = trainedNMF(components_=tr_I_f[:, lu_I], n_components=K, init='random', random_state=0, max_iter=1000,
                      beta_loss=loss, solver='mu')

ts_U_f = lu_U_f = lu_model.transform(zi_lu_graph_sparse)
ts_I_f = tr_I_f[:, ts_I]
predictions = np.matmul(lu_U_f, ts_I_f)

zi_ts_graph = zero_index_sparse_graph(ts_graph, axis=0, convert=ts_U_zero_indexer)
zi_ts_graph = zero_index_sparse_graph(zi_ts_graph, axis=1, convert=ts_I_zero_indexer)
#
topk = torch.topk(torch.tensor(predictions), n_ts_I)
# 3. predict graph_test: evaluate

nDCG_score = nDCG(np.r_[0:ts_U.shape[0]], topk[1], zi_ts_graph[0])
m = 20
precision_score = precision_at_m(np.r_[0:ts_U.shape[0]], topk[1], zi_ts_graph[0], m=m)

print("nDCG Score for is {}".format(np.mean(nDCG_score)))
print("Precision at {} is {}".format(m, np.mean(precision_score)))
