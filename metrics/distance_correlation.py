import numpy as np
from scipy.spatial.distance import pdist, squareform

def distance_correlation(A, B):
    #https://en.wikipedia.org/wiki/Distance_correlation
    #Input
    # A: the first variable
    # B: the second variable
    # The numbers of samples in the two variables must be same.
    #Output
    # dcor: the distance correlation of the two samples

    n = A.shape[0]
    if B.shape[0] != A.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(A))
    b = squareform(pdist(B))
    T1 = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    T2 = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    #Use Equation 2 to calculate distance covariances.
    dcov_T1_T2 = (T1 * T2).sum() / float(n * n)
    dcov_T1_T1 = (T1 * T1).sum() / float(n * n)
    dcov_T2_T2 = (T2 * T2).sum() / float(n * n)

    #Equation 1 in the paper.
    dcor = np.sqrt(dcov_T1_T2) / np.sqrt(np.sqrt(dcov_T1_T1) * np.sqrt(dcov_T2_T2))
    return dcor