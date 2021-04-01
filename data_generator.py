#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import math

class context:
    def __init__(self, K, lb_fv, ub_fv, T, d, true_theta, fv = None):
        if fv is None:
            fv = np.random.uniform(lb_fv, ub_fv, (T, K, d))
        self.K = K  
        self.d = d
        self.ub = ub_fv
        self.lb = lb_fv
        self.T = T
        self.fv = fv
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.theta = true_theta
        
    def build_bandit(self):
        for t in range(self.T):
            self.reward[t] = [self.fv[t][i].dot(self.theta) for i in range(self.K)] 
            max_rew, min_rew = max(self.reward[t]), min(self.reward[t])
            self.reward[t] = (self.reward[t] - min_rew) / 2 / (max_rew - min_rew)
            self.optimal[t] = max(self.reward[t])  # max reward
            
    def random_sample(self, t, i):
        # return np.random.uniform(self.reward[t][i], 0.5)
        return np.random.uniform(0, 2*self.reward[t][i])
    
    
class context_logistic:
    def __init__(self, K, lb_fv, ub_fv, T, d, true_theta, fv = None):
        if fv is None:
            fv = np.random.uniform(lb_fv, ub_fv, (T, K, d))
        self.K = K  
        self.d = d
        self.ub = ub_fv
        self.lb = lb_fv
        self.T = T
        self.fv = fv
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.theta = true_theta
        
    def logistic(self, x):
        return 1/(1+np.exp(-x))
    
    def build_bandit(self):
        for t in range(self.T):
            self.reward[t] = [self.logistic(self.fv[t][i].dot(self.theta)) for i in range(self.K)] 
            self.optimal[t] = max(self.reward[t])  # max reward
            
    def random_sample(self, t, i):
        return np.random.binomial(1, self.reward[t][i])

class covtype_random_feature:
    def __init__(self, reward, data, T, dim):
        self.d = dim
        self.T = T
        self.fv = []
        self.reward = [reward for t in range(self.T)]
        self.optimal = [max(reward) for t in range(self.T)]
        self.data = data
        self.y = []
    def build_bandit(self):
        for t in range(self.T):
            tmp = [None for _ in range(32)]
            tmpy = [0]*32
            idx = self.data[2]
            for i in range(32):
                data_idx = np.random.choice(idx[i])
                tmp[i] = self.data[0][data_idx]
                tmpy[i] = self.data[1][data_idx]
            self.fv.append(np.array(tmp))
            self.y.append(tmpy)
    def random_sample(self, t, i):
        return int(self.y[t][i])

        
class covtype:
    def __init__(self, reward, fv, T, dim = 10):
        self.d = dim
        self.T = T
        self.fv = [fv for t in range(self.T)] 
        self.reward = [reward for t in range(self.T)]
        self.optimal = [max(reward) for t in range(self.T)]  
    def random_sample(self, t, i):
        return np.random.binomial(1, self.reward[t][i])
    
class covtype_linear:
    def __init__(self, reward, fv, T, dim = 10):
        self.d = dim
        self.T = T
        self.fv = [fv for t in range(self.T)] 
        self.reward = [reward for t in range(self.T)]
        self.optimal = [max(reward) for t in range(self.T)]  
    def random_sample(self, t, i):
        # return np.random.normal(self.reward[t][i], 0.1)
        return np.random.beta(5, (5-5*self.reward[t][i]) / self.reward[t][i])
        # return np.random.binomial(1, self.reward[t][i])
    
class movielens:
    def __init__(self, T, true_theta, K = 30, d = 50, fv = None):
        num_movie = len(fv)
        self.fv = np.zeros((T, K, d))
        for t in range(T):
            idx = np.random.choice(num_movie, K, replace = False)
            self.fv[t] = fv[idx, :]
            # norms = [np.linalg.norm(self.fv[t][a]) for a in range(K)]
            # self.fv[t] /= np.max(norms)
            
        self.K = K  
        self.d = d
        self.T = T
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.theta = true_theta # / np.linalg.norm(true_theta)
        
    def build_bandit(self):
        min_rew = float('Inf')
        for t in range(self.T):
            self.reward[t] = [self.fv[t][i].dot(self.theta) for i in range(self.K)] 
            # make sure all rewards are within [0,1]
            max_rew, min_rew = max(self.reward[t]), min(self.reward[t])
            self.reward[t] = (self.reward[t] - min_rew) / (max_rew - min_rew) 
            
            # self.reward[t] = 0.01 + (self.reward[t] - min_rew) / (max_rew - min_rew) * 0.9
            # self.reward[t] -= min_rew
            self.optimal[t] = max(self.reward[t])  # max reward
        # print('minr {}, maxr {}'.format(min_rew, np.max(self.optimal)))
        
    def random_sample(self, t, i):
        # return np.random.uniform(0, 2*self.reward[t][i])
        # return np.random.binomial(1, self.reward[t][i])
        # return np.random.beta(5, (5-5*self.reward[t][i]) / self.reward[t][i])
        return np.random.normal(self.reward[t][i], 0.1)
        # return self.reward[t][i]

    
