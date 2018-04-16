import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
import pickle
import torch

from helpers.misc import zero_index_sparse_graph

from data_split import edge_samp_split

from helpers import nDCG
# Load data
loc = "/Users/ekansh/repos/data/{}"
ds = "nyt/"

with open(loc.format(ds) + 'data.pkl') as f:
    data = pickle.load(f)


graph = (data[:, :2], data[:, 2])

graph, _ = zero_index_sparse_graph(graph, axis = 0)
graph, _ = zero_index_sparse_graph(graph, axis = 1)

U = np.unique(graph[0][:, 0])
nU = U.shape[0]

I = np.unique(graph[0][:, 1])
nI = I.shape[0]

# Split data
train, test = edge_samp_split(graph, 0.8)
train_sparse = csr_matrix((train[1], (train[0][:, 0], train[0][:, 1])), shape=(nU, nI))



## Train NMF
K = 10
model = trainedNMF(n_components=K, init='random', random_state=0)
user_features = model.fit_transform(train_sparse)
item_features = model.components_

#
ts_U = np.unique(test[0][:, 0])
zi_test, test_convert = zero_index_sparse_graph(test)
mask_edges = train[0][np.in1d(train[0][:, 0], ts_U)]
mask_edges[:, 0] = test_convert[mask_edges[:, 0]]
test_user_features = user_features[ts_U]
predictions = np.matmul(test_user_features, item_features)

# Recommend top_k
K = I.shape[0]
topk = torch.topk(torch.tensor(predictions), K)

# Evaluate: More metrics?
score = nDCG(np.r_[0:ts_U.shape[0]], topk[1], zi_test[0])

