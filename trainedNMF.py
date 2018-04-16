from sklearn.decomposition.nmf import *


class trainedNMF(NMF):
    """
    light wrapper over NMF to load pretrained NMF.components_
    """
    def __init__(self, components_=None, n_components=None, init=None, solver='cd',
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None, alpha=0., l1_ratio=0., verbose=0,
                 shuffle=False):
        super(trainedNMF, self).__init__(n_components=n_components, init=init, solver=solver,
                                         beta_loss=beta_loss, tol=tol, max_iter=max_iter,
                                         random_state=random_state, alpha=alpha, l1_ratio=l1_ratio, verbose=verbose,
                                         shuffle=shuffle)
        self.components_ = components_
        self.n_components_ = components_.shape[0]


