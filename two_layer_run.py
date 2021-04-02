import numpy as np
import random
import math
import os
import argparse
import importlib

# from algorithms.data_generator import *
from twolayer.LinUCB import *
from twolayer.LinTS import *
from twolayer.UCB_GLM import *
from twolayer.AutoTuning import *
data_generator = importlib.import_module('twolayer.data_generator')

parser = argparse.ArgumentParser(description='simulations')
parser.add_argument('-split', '--split', type=float, default = 0.5, help = 'split size of candidate exploration rates')
parser.add_argument('-rep', '--rep', type=int, default = 10, help = 'repeat times')
parser.add_argument('-algo', '--algo', type=str, default = 'linucb', help = 'can also be lints, glmucb')
parser.add_argument('-model', '--model', type=str, default = 'linear', help = 'linear or logistic')
parser.add_argument('-k', '--k', type=int, default = 100, help = 'number of arms')
parser.add_argument('-max_rate', '--max_rate', type=float, default = -1, help = 'max explore rate')
parser.add_argument('-t', '--t', type=int, default = 10000, help = 'total time')
parser.add_argument('-d', '--d', type=int, default = 5, help = 'dimension')
parser.add_argument('-data', '--data', type=str, default = 'simulations', help = 'can be netflix or movielens')
parser.add_argument('-lamda', '--lamda', type=float, default = 1, help = 'lambda, regularization parameter')
parser.add_argument('-delta', '--delta', type=float, default = 0.1, help = 'error probability')
args = parser.parse_args()


explore_interval_length = args.split
T = args.t
d = args.d
rep = args.rep
algo = args.algo
model = args.model
K = args.k
lamda = args.lamda
delta = args.delta
datatype = args.data

if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + datatype + '/'):
    os.mkdir('results/' + datatype + '/')
if not os.path.exists('results/' + datatype + '/' + algo + '/'):
    os.mkdir('results/' + datatype + '/' + algo + '/')
path = 'results/' + datatype + '/' + algo + '/'

if datatype == 'movielens' or datatype == 'netflix':
    # check real data files exist:
    if not os.path.isfile('data/{}_users_matrix_d{}'.format(datatype, d)) or not os.path.isfile('data/{}_movies_matrix_d{}'.format(datatype, d)):
        print("{holder} data does not exist, will run preprocessing for {holder} data now. If you are running experiments for netflix data, then preprocessing might take a long time".format(holder=datatype))
        from data.preprocess_data import *
        process = eval("process_{}_data".format(datatype))
        process(d)
        print("real data processing done")   
    
    users = np.loadtxt("data/{}_users_matrix_d{}".format(datatype, d))
    fv = np.loadtxt("data/{}_movies_matrix_d{}".format(datatype, d))
    np.random.seed(0)
    thetas = np.zeros((rep, d))
    print(users.shape, fv.shape)
    for i in range(rep):
        thetas[i,:] = np.mean(users[np.random.choice(len(users), 100, replace = False), :], axis=0)

ub = 1/math.sqrt(d)
lb = -1/math.sqrt(d)

min_rate = 0
max_rate = 0.01*math.sqrt( d*math.log((T/lamda+1)/delta) ) + math.sqrt(lamda) + explore_interval_length
if args.max_rate != -1:
    max_rate = args.max_rate
J = np.arange(min_rate, max_rate, explore_interval_length)
lamdas = [1, 0.5, 0.1]
print("candidate set {}".format(J))

reg_theory = np.zeros((len(lamdas),T))
reg_auto = np.zeros(T)
reg_op = np.zeros(T)

methods = {
    'theory': '_theoretical_explore',
    'auto': '_auto',
    'op': '_op',
}
    
for i in range(rep):
    print(i, ": ", end = " ")
    np.random.seed(i+1)
    if 'lin' in algo:
        if datatype == 'simulations':
            theta = np.random.uniform(lb, ub, d)
            fv = np.random.uniform(lb, ub, (T, K, d))
            context = data_generator.context
        elif datatype in ['movielens', 'netflix']:
            theta = thetas[i, :]
            context = data_generator.movie
        bandit = context(K, T, d, true_theta = theta, fv=fv)
    elif 'glm' in algo:
        theta = np.random.uniform(lb, ub, d)
        fv = np.random.uniform(lb, ub, (T, K, d))
        context_logistic = data_generator.context_logistic
        bandit = context_logistic(K, lb, ub, T, d, true_theta = theta, fv=fv)
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
    result = {}
    for i,lamda in enumerate(lamdas):
        reg_theory[i,:] += fcts['theory'](lamda, delta)
        result['theory_lambda={}'.format(lamda)] = reg_theory[i,:]/(i+1)
        print('theory_lambda={}: {}'.format(lamda, reg_theory[i,-1]), end = " ")
        
    reg_op += fcts['op'](J, lamdas)
    result['op'] = reg_op/(i+1)
    
    reg_auto += fcts['auto'](J, lamdas)
    result['auto'] = reg_auto/(i+1)
    
    print("auto {}, op {}".format(reg_auto[-1], reg_op[-1]))
    for k,v in result.items():
        np.savetxt(path + k, v)   