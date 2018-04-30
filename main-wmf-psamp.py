import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
import pickle
import torch
import argparse

from helpers.misc import zero_index, zero_index_sparse_graph

from data_split import pq_samp_split

from helpers import nDCG, precision_at_m
import wmf

TINY = 1e-6

def main(data, p, q, K, M, num_iterations, alpha, lambda_reg, init_std):
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
    # Data Split
    tr_graph, lu_graph, ts_graph, tr_U, lu_I = pq_samp_split(graph, p, q)
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

    ## Train NMF
    # K = 10
    # print("Using {} loss".format(loss))
    # model = NMF(n_components=K, init='random', random_state=0, beta_loss=loss, solver='mu', max_iter=1000)
    zi_tr_graph = zero_index_sparse_graph(tr_graph, axis=0, convert=tr_U_zero_indexer)
    zi_tr_graph_sparse = csr_matrix((zi_tr_graph[1], (zi_tr_graph[0][:, 0], zi_tr_graph[0][:, 1])), shape=(n_tr_U, n_tr_I))


    S_tr_sparse = wmf.log_surplus_confidence_matrix(zi_tr_graph_sparse, alpha=alpha, epsilon=TINY)
    tr_U_f, tr_I_f = wmf.factorize(S_tr_sparse, num_factors=K, lambda_reg=lambda_reg, num_iterations=num_iterations, 
                                                init_std=init_std, verbose=True, dtype='float32', 
                                                recompute_factors=wmf.recompute_factors_bias)
    tr_I_f = tr_I_f.T
    # Train lookup model with item features fixed
    # 2. Train on graph_lookup: fix item_feat
    zi_lu_graph = zero_index_sparse_graph(lu_graph, axis=1, convert=lu_I_zero_indexer)
    zi_lu_graph = zero_index_sparse_graph(zi_lu_graph, axis=0, convert=lu_U_zero_indexer)
    zi_lu_graph_sparse = csr_matrix((zi_lu_graph[1], (zi_lu_graph[0][:, 0], zi_lu_graph[0][:, 1])), shape=(n_lu_U, n_lu_I))

    S_lu_sparse = wmf.log_surplus_confidence_matrix(zi_lu_graph_sparse, alpha=alpha, epsilon=TINY)
    lu_U_f, _ = wmf.factorize(zi_lu_graph_sparse, num_factors=K, lambda_reg=lambda_reg, num_iterations=num_iterations, 
                                                init_std=init_std, verbose=True, dtype='float32', 
                                                recompute_factors=wmf.recompute_factors_bias, V=tr_I_f[:, lu_I].T)

    ts_U_f = lu_U_f 
    ts_I_f = tr_I_f[:, ts_I]
    predictions = np.matmul(ts_U_f, ts_I_f)

    zi_ts_graph = zero_index_sparse_graph(ts_graph, axis=0, convert=ts_U_zero_indexer)
    zi_ts_graph = zero_index_sparse_graph(zi_ts_graph, axis=1, convert=ts_I_zero_indexer)
#
    topk = torch.topk(torch.tensor(predictions), n_ts_I)

    nDCG_score = nDCG(np.r_[0:ts_U.shape[0]], topk[1], zi_ts_graph[0])
    precision_score = precision_at_m(np.r_[0:ts_U.shape[0]], topk[1], zi_ts_graph[0], m=M)

    print("nDCG Score for is {}".format(np.mean(nDCG_score)))
    print("Precision at {} is {}".format(M, np.mean(precision_score)))
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("ds", help="Name of the dataset", type=str)

    # Optional arguments
    parser.add_argument("-p", "--p", type=int, help="p: Default=0.8", default = 0.8)
    parser.add_argument("-q", "--q", type=int, help="q: Default=0.8", default = 0.8)
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

    main(data, args.p, args.q, args.K, args.M, args.itr, args.alpha, args.lambda_reg, args.init_std)
