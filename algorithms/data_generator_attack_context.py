import numpy as np
import random
import math

class context:
    def __init__(self, C, K, lb_fv, ub_fv, T, d, true_theta, fv = None):
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
        self.attack_budget = C
        self.v = 0.5
        self.max_norm = 1
        
    def build_bandit(self):
        for t in range(self.T):
            self.reward[t] = np.array([self.fv[t][i].dot(self.theta) for i in range(self.K)])
            self.reward[t] = self.v + (self.reward[t] + 1) / 2 * (1-self.v)
            self.optimal[t] = max(self.reward[t])

    def random_reward(self, t, i):
        return np.random.normal(self.reward[t][i], 0.01)

    def context_strongest(self, t, i):
        target_index = np.argmin(self.reward[t])
        if i == target_index: # if the pulled arm is the target arm
            return self.fv[t], 0
        else:
            a = min(np.abs(2/self.v), self.max_norm/np.linalg.norm(self.fv[t][i])) # clip norm
            self.fv[t][i] = a * self.fv[t][i]
            cost = np.abs(a-1) * np.linalg.norm(self.fv[t][i])
            return self.fv[t], cost
        
    def context_topN(self, t, i):
        N = int(0.5*self.K)
        topN_index = [x[0] for x in sorted(enumerate(self.reward[t]), key=lambda x: x[1])[-N:]]
        if i not in topN_index:
            return self.fv[t], 0
        else:
            a = min(np.abs(2/self.v), self.max_norm/np.linalg.norm(self.fv[t][i]))
            self.fv[t][i] = a * self.fv[t][i]
            cost = np.abs(a-1) * np.linalg.norm(self.fv[t][i])
            return self.fv[t], cost

class movie:
    def __init__(self, C, T, true_theta, K = 20, d = 10, fv = None):
        num_movie = len(fv)
        self.fv = np.zeros((T, K, d))
        self.max_norm = float('-Inf')
        idx = np.random.choice(num_movie, K)
        for t in range(T):
            self.fv[t] = fv[idx, :]
            cur_max_norm = np.max( [np.linalg.norm(feature) for feature in self.fv[t]] )
            self.max_norm = max(self.max_norm, cur_max_norm)
            
        self.K = K  
        self.d = d
        self.T = T
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.theta = true_theta
        self.attack_budget = C
        self.v = 0.5
        
    def build_bandit(self):
        maxr = float('-Inf')
        minr = float('Inf')
        for t in range(self.T):
            self.reward[t] = np.array([self.fv[t][i].dot(self.theta) for i in range(self.K)])
            maxr = max(maxr, np.max(self.reward[t]))
            minr = min(minr, np.min(self.reward[t]))
        # make sure rewards are within 0 to 1
        for t in range(self.T):
            self.reward[t] = self.v + (1-self.v) * (self.reward[t] - minr) / (maxr - minr)
            self.optimal[t] = max(self.reward[t])

    def random_reward(self, t, i):
        return np.random.normal(self.reward[t][i], 0.01)
    
    def context_strongest(self, t, i):
        target_index = np.argmin(self.reward[t])
        if i == target_index: # if the pulled arm is the target arm
            return self.fv[t], 0
        else:
            a = np.abs(2/self.v)
            self.fv[t][i] = a * self.fv[t][i]
            cost = np.abs(a-1) * np.linalg.norm(self.fv[t][i])
            return self.fv[t], cost
        
    def context_topN(self, t, i):
        N = int(0.5*self.K)
        topN_index = [x[0] for x in sorted(enumerate(self.reward[t]), key=lambda x: x[1])[-N:]]
        if i not in topN_index:
            return self.fv[t], 0
        else:
            a = np.abs(2/self.v)
            self.fv[t][i] = a * self.fv[t][i]
            cost = np.abs(a-1) * np.linalg.norm(self.fv[t][i])
            return self.fv[t], cost