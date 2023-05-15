
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import numpy as np
import tqdm

def PairwiseRedundancy(x):
    h = []
    for i in tqdm.tqdm(range(x.size(0))):
        h.append(np.expand_dims(mutual_info_regression(x.T, x[i,:]), axis = 0 ))
    H = np.concatenate(h,axis = 0)

    return H

def loss(x, y, px):
    f = []
    for i in range(y.size(1)):
        f.append(np.expand_dims(mutual_info_classif(x.T, y[:,i]), axis = 0 ))
    # COMPUTE F
    F = np.concatenate(f, axis = 0)
    # COMPUTE H
    H = PairwiseRedundancy(x)
    # GET LOSS
    xtHx =  np.matmul(np.expand_dims(px, axis = 1).T, np.matmul(H, np.expand_dims(px, axis = 1)))
    xtF = np.matmul(F, np.expand_dims(px, axis = 1)).sum(axis = 0)
    L = xtHx - xtF
    #print(F.shape)
    #print(H.shape)
    #print(xtHx.shape)
    #print(xtF.shape)

    #print(L.shape)

    return L
