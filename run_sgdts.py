import numpy as np
import random
import math
import os
import argparse
import importlib

# from algorithms.data_generator import *
from algorithms.LinUCB import *
from algorithms.LinTS import *
from algorithms.UCB_GLM import *
from algorithms.AutoTuning import *
data_generator = importlib.import_module('algorithms.data_generator')

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
parser.add_argument('-delta', '--delta', type=float, default = 0.1, help = 'error probability')
parser.add_argument('-sigma', '--sigma', type=float, default = 0.5, help = 'sub gaussian parameter')
parser.add_argument('-js', '--js', nargs = '+', default = [0,0.01,0.1,1,10], help = 'exploration rates')
parser.add_argument('-etas', '--etas', nargs = '+', default = [0.01, 0.05, 0.1, 0.5, 1, 5, 10], help = 'exploration rates')
args = parser.parse_args()

T = args.t
d = args.d
rep = args.rep
algo = args.algo
model = args.model
K = args.k
delta = args.delta
datatype = args.data
sigma = args.sigma
J = args.js
J = [float(js) for js in J]
etas = args.etas
etas = [float(eta) for eta in etas]
lamda = 0

paras = {
    'eta0': etas,
    'alpha1': J,
    'alpha2': J,

}
print('tuning set of explore1, explore2, step size are {}'.format(paras))

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

reg_theory = np.zeros(T)
reg_auto = np.zeros(T)
reg_op = np.zeros(T)
reg_auto_3layer = np.zeros(T)
reg_auto_combined = np.zeros(T)

methods = {
    'auto': '_auto',
    'op': '_op',
    'auto_combined': '_combined',
}

for i in range(rep):
    np.random.seed(i+1)
    if 'lin' in algo:
        if datatype == 'simulations':
            theta = np.random.uniform(lb, ub, d)
            fv = np.random.uniform(lb, ub, (T, K, d))
            context = data_generator.context
        elif datatype in ['movielens', 'netflix']:
            theta = thetas[i, :]
            context = data_generator.movie
        bandit = context(K, T, d, sigma, true_theta = theta, fv=fv)
    elif 'glm' in algo or 'sgd' in algo:
        if datatype == 'simulations':
            theta = np.random.uniform(lb, ub, d)
            fv = np.random.uniform(-1, 1, (T, K, d))
            context_logistic = data_generator.context_logistic
            bandit = context_logistic(K, -1, 1, T, d, sigma, true_theta = theta, fv=fv)
        elif datatype in ['movielens', 'netflix']:
            context_logistic = data_generator.movie_logistic
            theta = thetas[i, :]
            bandit = context_logistic(K, T, d, sigma, true_theta = theta, fv=fv)
    bandit.build_bandit()
    if i==0:
        theory_explore_rate = g1 = sigma*math.sqrt( d/2*math.log(1+2*T/d) + 2*math.log(T)) 
        print('theoretical exploration rate is {}'.format(theory_explore_rate))
    
    print(i, ": ", end = " ")
    if algo == 'sgdts':
        algo_class = SGD_TS(bandit, T)
    elif algo == 'lints':
        algo_class = LinTS(bandit, T)
    elif algo == 'glmucb':
        algo_class = UCB_GLM(bandit, T)
    elif algo == 'linucb':
        algo_class = LinUCB(bandit, T)
    
    fcts = {
        k: getattr(algo_class, algo+methods[k]) 
        for k,v in methods.items()
    }
    reg_op += fcts['op']( {'eta0': paras['eta0']} )
    reg_auto += fcts['auto']({'eta0': paras['eta0']})
    reg_syndicated += fcts['auto'](paras)
    reg_combined += fcts['combined'](paras)
    
    print("op {}, tl {}, syn {}, combined {}".format(
        reg_op[-1], reg_auto[-1], reg_syndicated[-1], reg_combined[-1]))
    
    result = {
        'auto': reg_auto/(i+1),
        'op': reg_op/(i+1),
        'auto_3layer': reg_syndicated/(i+1),
        'auto_combined': reg_combined/(i+1),
    }
    for k,v in result.items():
        np.savetxt(path + k, v)   