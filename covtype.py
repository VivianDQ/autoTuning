import numpy as np
import random
import math
import os
import argparse
import time
import gzip
import pickle

from algorithms.data_generator import *
from algorithms.UCB_GLM import *
from algorithms.AutoTuning import *
from get_covtype_data import *

import warnings
# silent the following warnings since that the step size in grid search set does not always offer convergence
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
# ignore the following warning since that sklearn logistic regression does not always converge on the data
# it might be because that logistic model is not suitable for the data, this is probably the case especially for real datasets
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

parser = argparse.ArgumentParser(description='covtype')
parser.add_argument('-d', '--d', type=int, default = 10, help = 'number of features, choice of 10 (not use categorical features), 55 (use cat)')
parser.add_argument('-t', '--t', type=int, default = 10000, help = 'number of rounds')
parser.add_argument('-center', '--center', type=int, default = 1, help = 'use centriods as features (1), random feature (0)')
parser.add_argument('-split', '--split', type=float, default = 0.5, help = 'split size of candidate exploration rates')
parser.add_argument('-rep', '--rep', type=int, default = 10, help = 'repeat times')
parser.add_argument('-algo', '--algo', type=str, default = 'glmucb', help = 'can also be lints, glmucb')
parser.add_argument('-model', '--model', type=str, default = 'logistic', help = 'linear or logistic')
parser.add_argument('-k', '--k', type=int, default = 32, help = 'number of arms')
parser.add_argument('-max_rate', '--max_rate', type=float, default = -1, help = 'max explore rate')
parser.add_argument('-delta', '--delta', type=float, default = 0.1, help = 'error prob')
parser.add_argument('-data', '--data', type=str, default = 'covtype', help = 'can be netflix or movielens')
parser.add_argument('-lamda', '--lamda', type=float, default = 1, help = 'lambda, regularization parameter')
parser.add_argument('-sigma', '--sigma', type=float, default = 0.01, help = 'sub gaussian para')
args = parser.parse_args()

explore_interval_length = args.split
center = args.center # if center == 1, use cluster centroid as features, if center == 0, use random features
T = args.t
d = args.d
rep = args.rep
algo = args.algo
model = args.model
K = 32
lamda = args.lamda
delta = args.delta
datatype = args.data
sigma = args.sigma

if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + datatype + '/'):
    os.mkdir('results/' + datatype + '/')
if not os.path.exists('results/' + datatype + '/' + algo + '/'):
    os.mkdir('results/' + datatype + '/' + algo + '/')
path = 'results/' + datatype + '/' + algo + '/'

# extract, centeralize, standardize and cluster cover type data
if not os.path.isfile('data/rewards_covtype10.txt') or not os.path.isfile('data/features_covtype10.txt') or not os.path.isfile('data/X_covtype10.txt') or not os.path.isfile('data/y_covtype10.txt') or not os.path.isfile('data/idx_covtype10.txt'):
    get_covtype_data(10, center = 1)
# if not os.path.isfile('data/rewards_covtype55.txt') or not os.path.isfile('data/X_covtype55.txt') or not os.path.isfile('data/y_covtype55.txt') or not os.path.isfile('data/idx_covtype55.txt'):
#     get_covtype_data(55, center = 0)
print('data processing done')

if d == 10:
    with open('data/rewards_covtype10.txt', 'rb') as f:
        rewards = pickle.load(f)
    with open('data/features_covtype10.txt', 'rb') as f:
        features = pickle.load(f)
    with open('data/X_covtype10.txt', 'rb') as f:
        X = pickle.load(f)
    with open('data/y_covtype10.txt', 'rb') as f:
        y = pickle.load(f)
    with open('data/idx_covtype10.txt', 'rb') as f:
        idx = pickle.load(f)
    bandit_data = (X, y, idx)
    d = X.shape[1]
    print('d {}, K {}, T {}'.format(d,K,T))
if d == 55:
    with open('data/rewards_covtype55.txt', 'rb') as f:
        rewards = pickle.load(f)
    with open('data/X_covtype55.txt', 'rb') as f:
        X = pickle.load(f)
    with open('data/y_covtype55.txt', 'rb') as f:
        y = pickle.load(f)
    with open('data/idx_covtype55.txt', 'rb') as f:
        idx = pickle.load(f)
    bandit_data = (X, y, idx)
    d = X.shape[1]
    print('d {}, K {}, T {}'.format(d,K,T))

reg_theory = np.zeros(T)
reg_auto = np.zeros(T)
reg_op = np.zeros(T)
reg_auto_3layer = np.zeros(T)

min_rate = 0
# max_rate = sigma*math.sqrt( d*math.log((T/lamda+1)/delta) ) + math.sqrt(lamda) + explore_interval_length
if args.max_rate != -1:
    max_rate = args.max_rate
# J = np.arange(min_rate, max_rate, explore_interval_length)
# lamdas = np.arange(0.1, 1.1, 0.1)
print("candidate set {}".format(J))

methods = {
    'theory': '_theoretical_explore',
    'auto': '_auto',
    'op': '_op',
    'auto_3layer': '_auto_3layer',
}

for i in range(rep):
    print(i, ": ", end = " ")
    np.random.seed(i+1)
    if center:
        bandit = covtype(rewards, features, T, d)
    else:
        bandit = covtype_random_feature(rewards, bandit_data, T, d)
        bandit.build_bandit()    
    max_rate = sigma*math.sqrt( d*math.log((T*bandit.max_norm**2/lamda+1)/delta) ) + math.sqrt(lamda)*bandit.max_norm + explore_interval_length
    J = np.arange(min_rate, max_rate, explore_interval_length)
    if i==0: print("candidate set {}".format(J))
    algo_class = UCB_GLM(bandit, T)
    
    fcts = {
        k: getattr(algo_class, algo+methods[k]) 
        for k,v in methods.items()
    }
    
    reg_theory += fcts['theory'](lamda, delta)
    reg_op += fcts['op'](J, lamda)
    reg_auto += fcts['auto'](J, lamda)
    reg_auto_3layer += fcts['auto_3layer'](J, lamdas)
    
    print("theory {}, auto {}, op {}, auto_3layer {}".format(
        reg_theory[-1], reg_auto[-1], reg_op[-1], reg_auto_3layer[-1]))
    
    result = {
        'theory': reg_theory/(i+1),
        'auto': reg_auto/(i+1),
        'op': reg_op/(i+1),
        'auto_3layer': reg_auto_3layer/(i+1),
    }
    for k,v in result.items():
        np.savetxt(path + k, v)   