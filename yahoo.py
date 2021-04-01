import numpy as np
import random
import math
import os
import argparse
import pickle 

from data_generator import *
from LinUCB import *
from LinUCB_oracle import *
from LinUCB_knowC import *
from BOB import *
from Greedy import *
from LinTS import *


print('start processing yahoo data')
# if not os.path.isfile('../data/rewards_yahoo.txt') or not os.path.isfile('../data/features_yahoo.txt'):
#     extract_data()
with open('../data/rewards_yahoo.txt', 'rb') as f:
    rewards = pickle.load(f)
with open('../data/features_yahoo.txt', 'rb') as f:
    features = pickle.load(f)

parser = argparse.ArgumentParser(description='experiments for yahoo data')
parser.add_argument('-rep', '--rep', type=int, default = 10, help = 'repeat times')      
parser.add_argument('-attack', '--attack', type=str, default = 'flip_theta', help = 'type of attacks')
parser.add_argument('-data', '--data', type=str, default = 'yahoo', help = 'can be simulations, covtype_random, covtype, yahoo')
args = parser.parse_args()

attack = args.attack
args = parser.parse_args()
rep = args.rep # number of times to repeat experiments

datatype = args.data

if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + datatype + '/'):
    os.mkdir('results/' + datatype + '/')
if not os.path.exists('results/' + datatype + '/' + attack + '/'):
    os.mkdir('results/' + datatype + '/' + attack + '/')
path = 'results/' + datatype + '/' + attack + '/'

T = len(features)
K = 20
d = 6
C = T**(1/3) # (t**(1/3))10*math.log(T) # attack budget

explore = math.sqrt( 0.5*math.log(2*T**2*K) )
reg_linucb = np.zeros(T)
reg_linucb_oracle = np.zeros(T)
reg_linucb_knowC = np.zeros(T)
reg_lints = np.zeros(T)
reg_greedy = np.zeros(T)
reg_bob = np.zeros(T)

for i in range(rep):
    print(i, ": ", end = " ")
    np.random.seed(i+1)
    bandit = yahoo(C, rewards, features, d)
    
    linucb = LinUCB(attack, bandit, T)
    reg_linucb = linucb.linucb(explore)
    
    bob = BOB(attack, bandit, T)
    reg_bob = bob.bob(explore)
    
    knowC = LinUCB_knowC(attack, bandit, T)
    reg_linucb_knowC = knowC.linucb_knowC(explore)
    
    oracle = LinUCB_oracle(attack, bandit, T)
    reg_linucb_oracle = oracle.linucb_oracle(explore)
    
    ts = LinTS(attack, bandit, T)
    reg_lints = ts.lints(explore)
    
    grdy = Greedy(attack, bandit, T)
    reg_greedy = grdy.greedy()
    
    print("linucb {}, bob {}, knowC, {}, oracle {}, ts {}, greedy {}".format(reg_linucb[-1], reg_bob[-1], reg_linucb_knowC[-1], reg_linucb_oracle[-1], reg_lints[-1], reg_greedy[-1]))
    
    with open(path + 'linucb','a') as f:
        np.savetxt(f, reg_linucb, fmt="%.5f", delimiter=' ', newline=' ')
        f.write('\n')
    with open(path + 'bob','a') as f:
        np.savetxt(f, reg_bob, fmt="%.5f", delimiter=' ', newline=' ')
        f.write('\n')
    with open(path + 'linucb_knowC','a') as f:
        np.savetxt(f, reg_linucb_knowC, fmt="%.5f", delimiter=' ', newline=' ')
        f.write('\n')
    with open(path + 'linucb_oracle','a') as f:
        np.savetxt(f, reg_linucb_oracle, fmt="%.5f", delimiter=' ', newline=' ')
        f.write('\n')
    with open(path + 'lints','a') as f:
        np.savetxt(f, reg_lints, fmt="%.5f", delimiter=' ', newline=' ')
        f.write('\n')   
    with open(path + 'greedy','a') as f:
        np.savetxt(f, reg_greedy, fmt="%.5f", delimiter=' ', newline=' ')
        f.write('\n')

    