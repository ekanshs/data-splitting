import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
import pickle
import torch
import argparse

from helpers.misc import zero_index_sparse_graph

from data_split import edge_samp_split

from helpers import nDCG, precision_at_m
import wmf

TINY = 1e-6

def main(data, p, K, M, num_iterations, alpha, lambda_reg, init_std):
    # loc = "/Users/ekansh/repos/data/{}"
    # ds = "nyt/"
    # loss = 'kullback-leibler'
    # # loss = "frobenius"

    graph = (data[:, :2], data[:, 2])

    graph, _ = zero_index_sparse_graph(graph, axis = 0)
    graph, _ = zero_index_sparse_graph(graph, axis = 1)

    U = np.unique(graph[0][:, 0])
    nU = U.shape[0]

    I = np.unique(graph[0][:, 1])
    nI = I.shape[0]

    # Split data
    train, test = edge_samp_split(graph, p)
    train_sparse = csr_matrix((train[1], (train[0][:, 0], train[0][:, 1])), shape=(nU, nI))

    ## Train NMF
    # K = 10
    # print("Using {} loss".format(loss))
    # model = NMF(n_components=K, init='random', random_state=0, beta_loss=loss, solver='mu', max_iter=1000)
    S = wmf.log_surplus_confidence_matrix(train_sparse, alpha=alpha, epsilon=TINY)

    user_features, item_features = wmf.factorize(S, num_factors=K, lambda_reg=lambda_reg, num_iterations=num_iterations, 
                                                init_std=init_std, verbose=True, dtype='float32', 
                                                recompute_factors=wmf.recompute_factors_bias)

    ts_U = np.unique(test[0][:, 0])
    zi_test, test_convert = zero_index_sparse_graph(test)
    mask_edges = train[0][np.in1d(train[0][:, 0], ts_U)]
    mask_edges[:, 0] = test_convert[mask_edges[:, 0]]

    test_user_features = user_features[ts_U]
    predictions = np.matmul(test_user_features, item_features.T)

    # Scatter update might be faster but for correctness
    for edge in mask_edges:
        predictions[tuple(edge)] = 0.

    # Recommend top_k

    topk = torch.topk(torch.tensor(predictions), I.shape[0])[1].numpy()

    # Evaluate: More metrics?
    nDCG_score = nDCG(np.r_[0:ts_U.shape[0]], topk, zi_test[0])

    # m=20

    precision_score = precision_at_m(np.r_[0:ts_U.shape[0]], topk, zi_test[0], M)


    print("nDCG Score for is {}".format(np.mean(nDCG_score)))
    print("Precision at {} is {}".format(m, np.mean(precision_score)))
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("ds", help="Name of the dataset", type=str)

    # Optional arguments
    parser.add_argument("-p", "--p", type=int, help="Split p: Default = 0.8", default = 0.8)
    parser.add_argument("-K", "--K", type=int, help="Number of communities: default K = 30", default = 30)
    parser.add_argument("-M", "--M", type=int, help="TopM precision", default = 20)
    parser.add_argument("-itr", "--itr", type=int, help="Number of iterations; default itr = 400", default = 40)
    parser.add_argument("-a", "--alpha", type=float, help="alpha; default 2.0", default = 2.0)
    parser.add_argument("-l", "--lambda_reg", type=float, help="Lambda Reg: default l = 1e-5", default = 1e-5)
    parser.add_argument("-i", "--init_std", type=float, help="Init std: default l = 1e-5", default = 0.01)
    
    args = parser.parse_args()
    loc = "/Users/ekansh/repos/data/{}/"

    print("Dataset is : {}".format(args.ds))
    with open(loc.format(args.ds) + 'data.pkl') as f:
        data = pickle.load(f)

    main(data, args.p, args.K, args.M, args.itr, args.alpha, args.lambda_reg, args.init_std)
