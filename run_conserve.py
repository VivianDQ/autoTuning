import numpy as np
import random
import math
import os
import argparse
import importlib
from scipy.stats import truncnorm

# from algorithms.data_generator import *
from algorithms.LinUCB import *
from algorithms.LinTS import *
from algorithms.UCB_GLM import *
from algorithms.AutoTuning import *
data_generator = importlib.import_module('algorithms.data_generator')

parser = argparse.ArgumentParser(description='simulations')
parser.add_argument('-split', '--split', type=float, default = 0.5, help = 'split size of candidate exploration rates')
parser.add_argument('-rep', '--rep', type=int, default = 5, help = 'repeat times')
parser.add_argument('-algo', '--algo', type=str, default = 'linucb', help = 'can also be lints, glmucb')
parser.add_argument('-model', '--model', type=str, default = 'linear', help = 'linear or logistic')
parser.add_argument('-k', '--k', type=int, default = 100, help = 'number of arms')
parser.add_argument('-max_rate', '--max_rate', type=float, default = 10, help = 'max explore rate')
parser.add_argument('-t', '--t', type=int, default = 10000, help = 'total time')
parser.add_argument('-d', '--d', type=int, default = 5, help = 'dimension')
parser.add_argument('-dist', '--dist', type=str, default = 'uniform', help = 'can also be normal')
parser.add_argument('-data', '--data', type=str, default = 'grid_all', help = 'grid search')
parser.add_argument('-lamda', '--lamda', type=float, default = 1, help = 'lambda, regularization parameter')
parser.add_argument('-delta', '--delta', type=float, default = 0.1, help = 'error probability')
parser.add_argument('-sigma', '--sigma', type=float, default = 0.01, help = 'sub gaussian parameter')
parser.add_argument('-lamdas', '--lamdas', type=list, default = [0.1,0.5,1], help = 'lambdas')
args = parser.parse_args()

explore_interval_length = args.split
dist = args.dist
T = args.t
d = args.d
rep = args.rep
algo = args.algo
model = args.model
K = args.k
lamda = args.lamda
delta = args.delta
datatype = args.data
sigma = args.sigma
lamdas = args.lamdas

if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + datatype + '/'):
    os.mkdir('results/' + datatype + '/')
if not os.path.exists('results/' + datatype + '/' + algo + '/'):
    os.mkdir('results/' + datatype + '/' + algo + '/')
path = 'results/' + datatype + '/' + algo + '/'

ub = 1/math.sqrt(d)
lb = -1/math.sqrt(d)

reg_theory = np.zeros(T)
min_rate = 0
max_rate = args.max_rate
J = np.arange(min_rate, max_rate + explore_interval_length, explore_interval_length)
final = {k:[] for k in J}

for explore in J:
    reg_theory = np.zeros(T)
    for i in range(rep):
        # print(i, ": ", end = " ")
        np.random.seed(i+1)
        theta = np.random.uniform(lb, ub, d)
        if 'lin' in algo:
            if dist == 'uniform':
                fv = np.random.uniform(lb, ub, (T, K, d))
            elif dist == 'uniform_fixed':
                fv = [np.random.uniform(lb, ub, (K, d))]*T
            elif dist == 'normal':
                fv = truncnorm.rvs(lb, ub, size=(T,K,d))
            context = data_generator.context
            bandit = context(K, T, d, sigma, true_theta = theta, fv=fv)
        elif 'glm' in algo:
            if dist == 'uniform':
                fv = np.random.uniform(lb, ub, (T, K, d))
            elif dist == 'uniform_fixed':
                fv = [np.random.uniform(lb, ub, (K, d))]*T
            elif dist == 'normal':
                fv = truncnorm.rvs(lb, ub, size=(T,K,d))
            context_logistic = data_generator.context_logistic
            bandit = context_logistic(K, lb, ub, T, d, sigma, true_theta = theta, fv=fv)
        bandit.build_bandit()

        if algo == 'linucb':
            algo_class = LinUCB(bandit, T)
        elif algo == 'lints':
            algo_class = LinTS(bandit, T)
        elif algo == 'glmucb':
            algo_class = UCB_GLM(bandit, T)

        fcts = getattr(algo_class, algo+'_theoretical_explore')
        reg_theory += fcts(lamda, delta, explore)
    final[explore] = reg_theory[-1]/rep
    print("explore = {} done!".format(explore))
    
theory = np.zeros(T)
for i in range(rep):
    # print(i, ": ", end = " ")
    np.random.seed(i+1)
    theta = np.random.uniform(lb, ub, d)
    if 'lin' in algo:
        if dist == 'uniform':
            fv = np.random.uniform(lb, ub, (T, K, d))
        elif dist == 'uniform_fixed':
            fv = [np.random.uniform(lb, ub, (K, d))]*T
        elif dist == 'normal':
            fv = truncnorm.rvs(lb, ub, size=(T,K,d))
        context = data_generator.context
        bandit = context(K, T, d, sigma, true_theta = theta, fv=fv)
    elif 'glm' in algo:
        if dist == 'uniform':
            fv = np.random.uniform(lb, ub, (T, K, d))
        elif dist == 'uniform_fixed':
            fv = [np.random.uniform(lb, ub, (K, d))]*T
        elif dist == 'normal':
            fv = truncnorm.rvs(lb, ub, size=(T,K,d))
        context_logistic = data_generator.context_logistic
        bandit = context_logistic(K, lb, ub, T, d, sigma, true_theta = theta, fv=fv)
    bandit.build_bandit()

    if algo == 'linucb':
        algo_class = LinUCB(bandit, T)
    elif algo == 'lints':
        algo_class = LinTS(bandit, T)
    elif algo == 'glmucb':
        algo_class = UCB_GLM(bandit, T)

    fcts = getattr(algo_class, algo+'_theoretical_explore')
    theory += fcts(lamda, delta, -1)
final['theory'] = theory / rep

print(final)
    
# final = np.array([ [k] + v for k,v in final.items() ])       
np.savetxt(path + dist, final)   