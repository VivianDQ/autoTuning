#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import math
import os
import argparse
import time
import gzip
import pickle

from data_generator import *
from LinUCB import *
from LinTS import *
from UCB_GLM import *
from AutoTuning import *
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
parser.add_argument('-center', '--center', type=int, default = 1, help = 'use centriods as features (1), random feature (0)')
parser.add_argument('-split', '--split', type=float, default = 0.5, help = 'split size of candidate exploration rates')
parser.add_argument('-rep', '--rep', type=int, default = 10, help = 'repeat times')
parser.add_argument('-algo', '--algo', type=str, default = 'glmucb', help = 'can also be lints, glmucb')
parser.add_argument('-model', '--model', type=str, default = 'logistic', help = 'linear or logistic')
parser.add_argument('-k', '--k', type=int, default = 32, help = 'number of arms')
parser.add_argument('-max_rate', '--max_rate', type=float, default = 5, help = 'max explore rate')
args = parser.parse_args()


explore_interval_length = args.split
T = 10**3

algo = args.algo
rep = args.rep  # repeat times, set to 10
d = args.d  # feature dimension, if use only quantitative features, d = 10, otherwise, d = 55
center = args.center # if center == 1, use cluster centroid as features, if center == 0, use random features
K = 32


# extract, centeralize, standardize and cluster cover type data
if not os.path.isfile('../data/rewards_covtype10.txt') or not os.path.isfile('../data/features_covtype10.txt') or not os.path.isfile('../data/X_covtype10.txt') or not os.path.isfile('../data/y_covtype10.txt') or not os.path.isfile('../data/idx_covtype10.txt'):
    get_covtype_data(10, center = 1)
if not os.path.isfile('../data/rewards_covtype55.txt') or not os.path.isfile('../data/X_covtype55.txt') or not os.path.isfile('../data/y_covtype55.txt') or not os.path.isfile('../data/idx_covtype55.txt'):
    get_covtype_data(55, center = 0)
print('data processing done')

if d == 10:
    with open('../data/rewards_covtype10.txt', 'rb') as f:
        rewards = pickle.load(f)
    with open('../data/features_covtype10.txt', 'rb') as f:
        features = pickle.load(f)
    with open('../data/X_covtype10.txt', 'rb') as f:
        X = pickle.load(f)
    with open('../data/y_covtype10.txt', 'rb') as f:
        y = pickle.load(f)
    with open('../data/idx_covtype10.txt', 'rb') as f:
        idx = pickle.load(f)
    bandit_data = (X, y, idx)
    d = X.shape[1]
    print('d {}, K {}, T {}'.format(d,K,T))
if d == 55:
    with open('../data/rewards_covtype55.txt', 'rb') as f:
        rewards = pickle.load(f)
    with open('../data/X_covtype55.txt', 'rb') as f:
        X = pickle.load(f)
    with open('../data/y_covtype55.txt', 'rb') as f:
        y = pickle.load(f)
    with open('../data/idx_covtype55.txt', 'rb') as f:
        idx = pickle.load(f)
    bandit_data = (X, y, idx)
    d = X.shape[1]
    print('d {}, K {}, T {}'.format(d,K,T))

if center:
    datatype = 'covtype'
else:
    datatype = 'covtype_random_feature'

reg_grid = np.zeros(T)
reg_theory = np.zeros(T)
reg_auto = np.zeros(T)
reg_auto_adv = np.zeros(T)
reg_fixed = np.zeros(T)
reg_op = np.zeros(T)

lamda = 1
min_rate = 0
# max_rate = 2 * int(math.sqrt( 0.5*math.log(2*T**2*K) )) 
if algo == 'linucb':
    max_rate = int(0.5 * math.sqrt(d * math.log(T**2+T) ) + math.sqrt(lamda)) + 1
    explore = 0.5 * math.sqrt(d * math.log(T**2+T)) + math.sqrt(lamda)  # math.sqrt( 0.5*math.log(2*T**2*K) )
elif algo == 'lints':
    max_rate = int(0.5 * math.sqrt(d * math.log(T**2+T) ) + math.sqrt(lamda)) + 1 
    explore = 0.5 * math.sqrt(d * math.log(T**2+T)) + math.sqrt(lamda) # 0.5 * math.sqrt( 9*d * math.log(T**2) )
elif algo == 'glmucb':
    max_rate = int(0.5 * math.sqrt(d/2*math.log( 1+2*T/d ) + math.log(T)) + math.sqrt(lamda))+1
    explore = 0.5 * math.sqrt(d/2*math.log( 1+2*T/d ) + math.log(T)) + math.sqrt(lamda) 
    # 0.5 * math.sqrt(d * math.log(T**2+T)) + math.sqrt(lamda)
    # max_rate = int(0.5 * math.sqrt(d * math.log(T**2+T) ) + math.sqrt(lamda)) + 1 
    # explore = 0.5 * math.sqrt(d * math.log(T**2+T)) + math.sqrt(lamda)

if args.max_rate != 5:
    max_rate = args.max_rate
J = np.arange(min_rate, max_rate, explore_interval_length)
print("candidate set {}".format(J))

grid = np.zeros((len(J),T))
methods = {
    'fixed': '_fixed_explore',
    'theory': '_theoretical_explore',
    'auto': '_auto',
    'auto_adv': '_auto_advanced',
    'op': '_op',
    'grid': '_fixed_explore',
}

for i in range(rep):
    print(i, ": ", end = " ")
    np.random.seed(i+1)
    if center:
        if 'lin' in algo:
            bandit = covtype_linear(rewards, features, T, d)
        elif 'glm' in algo:
            bandit = covtype(rewards, features, T, d)
    else:
        bandit = covtype_random_feature(rewards, bandit_data, T, d)
        bandit.build_bandit()
    
    if algo == 'linucb':
        algo_class = LinUCB(bandit, T)
    elif algo == 'lints':
        algo_class = LinTS(bandit, T)
    elif algo == 'glmucb':
        algo_class = UCB_GLM(bandit, T)
    
    fcts = {
        k: getattr(algo_class, algo+methods[k]) 
        for k,v in methods.items()
    }
    
    print('data done')
    reg_fixed += fcts['fixed'](explore)
    print('fixed done')

    reg_theory += fcts['theory'](explore)
    print('theory done')
    
    reg_op += fcts['op'](J)
    print('op done')
    
    reg_auto += fcts['auto'](J)
    print('auto done')
    
    # reg_auto_adv += fcts['auto_adv'](J)
    # print('auto adv done')

    # for j in range(len(J)):
    #     grid[j,:] += fcts['fixed'](j)

    print("fixed {}, theory {}, auto {}, op {}, auto_adv {}, grid {}".format(
        reg_fixed[-1], reg_theory[-1], reg_auto[-1], reg_op[-1], reg_auto_adv[-1], min(grid[:,-1]) ))
    

indexJ = np.argmin(grid[:,-1])
reg_grid = grid[indexJ,:]
result = {
    'fixed': reg_fixed/rep,
    'theory': reg_theory/rep,
    'auto': reg_auto/rep,
    'auto_adv': reg_auto_adv/rep,
    'op': reg_op/rep,
    'grid': reg_grid/rep,
}

if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + datatype + '/'):
    os.mkdir('results/' + datatype + '/')
if not os.path.exists('results/' + datatype + '/' + algo):
    os.mkdir('results/' + datatype + '/' + algo)
path = 'results/' + datatype + '/' + algo + '/'

    
for k,v in result.items():
    np.savetxt('results/' + datatype + '/' + algo + '/' + k, v)
