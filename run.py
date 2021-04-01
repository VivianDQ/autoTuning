import numpy as np
import random
import math
import os
import argparse

from data_generator import *
from LinUCB import *
from LinTS import *
from UCB_GLM import *
from AutoTuning import *


parser = argparse.ArgumentParser(description='simulations')
parser.add_argument('-split', '--split', type=float, default = 0.5, help = 'split size of candidate exploration rates')
parser.add_argument('-rep', '--rep', type=int, default = 10, help = 'repeat times')
parser.add_argument('-algo', '--algo', type=str, default = 'linucb', help = 'can also be lints, glmucb')
parser.add_argument('-model', '--model', type=str, default = 'linear', help = 'linear or logistic')
parser.add_argument('-k', '--k', type=int, default = 100, help = 'number of arms')
parser.add_argument('-max_rate', '--max_rate', type=float, default = 5, help = 'max explore rate')
args = parser.parse_args()


explore_interval_length = args.split
T = 10**4
d = 5
rep = args.rep
algo = args.algo
model = args.model
K = args.k
ub = 1/math.sqrt(d)
lb = -1/math.sqrt(d)


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
    # max_rate = int(0.5 * math.sqrt(d/2*math.log( 1+2*T/d ) + math.log(T)) ) + 1
    # explore = 0.5 * math.sqrt(d/2*math.log( 1+2*T/d ) + math.log(T)) # 0.5 * math.sqrt(d/2*math.log( 1+2*T/d ) + math.log(T))
    max_rate = int(0.5 * math.sqrt(d * math.log(T**2+T) ) + math.sqrt(lamda)) + 1 
    explore = 0.5 * math.sqrt(d * math.log(T**2+T)) + math.sqrt(lamda)

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
    # theta = np.random.uniform(lb, ub, d)
    # fv = np.random.uniform(lb, ub, (T, K, d))
    
    if 'lin' in algo:
        theta = np.random.uniform(lb, ub, d)
        fv = np.random.uniform(lb, ub, (T, K, d))
        bandit = context(K, lb, ub, T, d, true_theta = theta, fv=fv)
    elif 'glm' in algo:
        theta = np.random.uniform(lb, ub, d)
        fv = np.random.uniform(lb, ub, (T, K, d))
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
    reg_fixed += fcts['fixed'](explore)

    reg_theory += fcts['theory'](explore)

    reg_op += fcts['op'](J)
    
    reg_auto += fcts['auto'](J)
    
    # reg_auto_adv += fcts['auto_adv'](J)


    for j in range(len(J)):
        grid[j,:] += fcts['fixed'](j)

    print("fixed {}, theory {}, auto {}, op {}, auto_adv {}, grid {}".format(
        reg_fixed[-1], reg_theory[-1], reg_auto[-1], reg_op[-1], reg_auto_adv[-1], min(grid[:,-1]) ))
    
indexJ = np.argmin(grid[:,-1])
reg_grid = grid[indexJ,:]
result = {
    'fixed': reg_fixed/rep,
    'theory': reg_theory/rep,
    'auto': reg_auto/rep,
    # 'auto_adv': reg_auto_adv/rep,
    'op': reg_op/rep,
    'grid': reg_grid/rep,
    'grid_all': grid/rep,
}


name = 'simulation_d' + str(d) + '_k' + str(K)
if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + name + '/'):
    os.mkdir('results/' + name + '/')
if not os.path.exists('results/' + name + '/' + algo):
    os.mkdir('results/' + name + '/' + algo)
    
for k,v in result.items():
    np.savetxt('results/' + name + '/' + algo + '/' + k, v)
