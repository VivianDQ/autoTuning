import numpy as np
import random
import math

class context:
    def __init__(self, K, T, d, true_theta, fv = None):
        self.ub = 1/math.sqrt(d)
        self.lb = -1/math.sqrt(d)
        if fv is None:
            fv = np.random.uniform(self.lb, self.ub, (T, K, d))
        self.K = K  
        self.d = d
        self.T = T
        self.fv = fv
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.theta = true_theta
        self.max_norm = 1
        
    def build_bandit(self):
        for t in range(self.T):
            self.reward[t] = np.array([self.fv[t][i].dot(self.theta) for i in range(self.K)])
            # make sure all rewards are within [0,1]
            self.reward[t] = (self.reward[t] +1) / 2 # since x^T \theta lies in [-1,1]
            self.optimal[t] = max(self.reward[t])
            
    def random_sample(self, t, i):
        return np.random.normal(self.reward[t][i], 0.01)
    
    
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
        self.max_norm = math.sqrt(d*self.ub**2)
        
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

        
        
class movie:
    def __init__(self, K = 100, T = 10000, d = 5, true_theta = None, fv = None):
        num_movie = len(fv)
        self.fv = np.zeros((T, K, d))
        self.max_norm = float('-Inf')
        for t in range(T):
            idx = np.random.choice(num_movie, K, replace=False)
            self.fv[t] = fv[idx, :]
            cur_max_norm = np.max( [np.linalg.norm(feature) for feature in self.fv[t]] )
            self.max_norm = max(self.max_norm, cur_max_norm)
            
        self.K = K  
        self.d = d
        self.T = T
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.theta = true_theta
        
    def build_bandit(self):
        maxr = float('-Inf')
        minr = float('Inf')
        for t in range(self.T):
            self.reward[t] = np.array([self.fv[t][i].dot(self.theta) for i in range(self.K)])
            maxr = max(maxr, np.max(self.reward[t]))
            minr = min(minr, np.min(self.reward[t]))
        # make sure rewards are within 0 to 1
        # for t in range(self.T):
            # self.reward[t] = (self.reward[t] - minr) / (maxr - minr)
            self.optimal[t] = max(self.reward[t])

    def random_sample(self, t, i):
        return np.random.normal(self.reward[t][i], 0.01)