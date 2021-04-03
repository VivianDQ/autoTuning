import numpy as np
import time
import gzip
import math
import pickle
import random
from sklearn.cluster import KMeans
from sklearn import preprocessing

# extract, centeralize, standardize and cluster cover type data
def get_covtype_data(d, center = 0):
    if center == 1:
        print('use cluster centroid as features, d = {},'.format(d), 'start processing data')
    if center == 0:
        print('use random features, d = {},'.format(d), 'start processing data')

    lines = []
    labels = []
    t0 = time.time()
    # save the 'covtype.data.gz' under the 'data' folder before running this code
    with gzip.open('../data/covtype.data.gz', "r") as f:
        for line in f:
            line = line.split(b',')
            tmp = line[:d]
            y = int(line[-1])
            if y!=1:
                y = 0
            x = [float(i) for i in tmp]
            lines += [x]
            labels += [y]

    X = np.array(lines)
    y = np.array(labels)
    X[:,:10] = preprocessing.scale(X[:,:10])

    np.random.seed(0)
    kmeans = KMeans(n_clusters=32, random_state=1).fit(X)
    rewards = [0]*32
    idx = [None for _ in range(32)]
    features = np.array(kmeans.cluster_centers_)
    for nc in range(32):
        idx[nc] = np.where(kmeans.labels_ == nc)[0]
        num, den = sum(y[idx[nc]]), len(idx[nc])
        rewards[nc] = num / den
    bandit_data = (X, y, idx)
    K, d = 32, X.shape[1]
    
    if center == 1:
        with open('../data/rewards_covtype10.txt', 'wb') as f:
            pickle.dump(rewards, f)
        with open('../data/features_covtype10.txt', 'wb') as f:
            pickle.dump(features, f)
        with open('../data/X_covtype10.txt', 'wb') as f:
            pickle.dump(X, f)
        with open('../data/y_covtype10.txt', 'wb') as f:
            pickle.dump(y, f)
        with open('../data/idx_covtype10.txt', 'wb') as f:
            pickle.dump(idx, f)
    else:
        with open('../data/rewards_covtype55.txt', 'wb') as f:
            pickle.dump(rewards, f)
        with open('../data/X_covtype55.txt', 'wb') as f:
            pickle.dump(X, f)
        with open('../data/y_covtype55.txt', 'wb') as f:
            pickle.dump(y, f)
        with open('../data/idx_covtype55.txt', 'wb') as f:
            pickle.dump(idx, f)
