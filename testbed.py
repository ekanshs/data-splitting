import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
import pickle
import torch

from helpers.misc import zero_index_sparse_graph

from data_split import edge_samp_split

from helpers import nDCG, precision_at_m
# Load data
loc = "/Users/ekansh/repos/data/{}"
ds = "nyt/"
# loss = 'kullback-leibler'
loss = "frobenius"

print("Dataset is : {}".format(ds))
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
print("Using {} loss".format(loss))
model = NMF(n_components=K, init='random', random_state=0, beta_loss=loss, solver='mu', max_iter=1000)
user_features = model.fit_transform(train_sparse)
item_features = torch.tensor(model.components_)

#
ts_U = np.unique(test[0][:, 0])
zi_test, test_convert = zero_index_sparse_graph(test)
mask_edges = train[0][np.in1d(train[0][:, 0], ts_U)]
mask_edges[:, 0] = test_convert[mask_edges[:, 0]]

test_user_features = user_features[ts_U]
predictions = np.matmul(test_user_features, item_features)

# Scatter update might be faster but for correctness
for edge in mask_edges:
    predictions[tuple(edge)] = 0.

# Recommend top_k

topk = torch.topk(torch.tensor(predictions), I.shape[0])[1].numpy()

# Evaluate: More metrics?
nDCG_score = nDCG(np.r_[0:ts_U.shape[0]], topk, zi_test[0])

m=20

precision_score = precision_at_m(np.r_[0:ts_U.shape[0]], topk, zi_test[0], m)


print("nDCG Score for is {}".format(np.mean(nDCG_score)))
print("Precision at {} is {}".format(m, np.mean(precision_score)))

