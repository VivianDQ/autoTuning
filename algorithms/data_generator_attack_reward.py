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
        self.max_norm = 1
        
    def build_bandit(self):
        for t in range(self.T):
            self.reward[t] = np.array([self.fv[t][i].dot(self.theta) for i in range(self.K)])
            # make sure all rewards are within [0,1]
            self.reward[t] = (self.reward[t] +1) / 2 # since x^T \theta lies in [-1,1]
            self.optimal[t] = max(self.reward[t])

    def random_reward(self, t, i):
        return np.random.normal(self.reward[t][i], 0.01)
    
    def garcelon_strongest(self, t, i):
        true = self.random_reward(t,i)
        # let target arm be the worst arm forever
        target = np.argmin(self.reward[t])
        # push everything to the min = 0
        attacked = np.random.normal(0, 0.01)
        cost = np.abs(true-attacked)
        if i == target:
            return true, 0
        else:
            return attacked, cost
        
    def oracle_strongest(self, t, i, eps = 0.01):
        true = self.random_reward(t,i)
        # let target arm be the worst arm forever
        target = np.argmin(self.reward[t])
        min_reward = min(self.reward[t])
        attacked = np.random.normal(min_reward-eps, 0.01)
        cost = np.abs(true-attacked)
        
        if i == target or true <= attacked:
            return true, 0
        else:
            return attacked, cost
    
    def garcelon_topN(self, t, i):
        N = int(0.5*self.K) # corrupt top half arms
        topN_index = [x[0] for x in sorted(enumerate(self.reward[t]), key=lambda x: x[1])[-N:]]
        true = self.random_reward(t,i)
        attacked = np.random.normal(0, 0.01)
        cost = np.abs(true-attacked)
        if i not in topN_index:
            return true, 0
        else:
            return attacked, cost
        
    def oracle_topN(self, t, i, eps=0.01):
        N = int(0.5*self.K) # corrupt top half arms
        topN_index = [x[0] for x in sorted(enumerate(self.reward[t]), key=lambda x: x[1])[-N:]]
        true = self.random_reward(t,i)
        min_reward = min(self.reward[t])
        attacked = np.random.normal(min_reward-eps, 0.01)
        cost = np.abs(true-attacked)
        
        if i not in topN_index or true <= attacked:
            return true, 0
        else:
            return attacked, cost
    
    def flip_theta(self, t, i):
        true = self.random_reward(t,i)
        attacked = np.random.normal(-self.reward[t][i], 0.01)
        cost = np.abs(true-attacked)
        return attacked, cost

    
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
        
    def build_bandit(self):
        maxr = float('-Inf')
        minr = float('Inf')
        for t in range(self.T):
            self.reward[t] = np.array([self.fv[t][i].dot(self.theta) for i in range(self.K)])
            maxr = max(maxr, np.max(self.reward[t]))
            minr = min(minr, np.min(self.reward[t]))
        # make sure rewards are within 0 to 1
        for t in range(self.T):
            self.reward[t] = (self.reward[t] - minr) / (maxr - minr)
            self.optimal[t] = max(self.reward[t])

    def random_reward(self, t, i):
        return np.random.normal(self.reward[t][i], 0.01)
    
    def garcelon_strongest(self, t, i):
        true = self.random_reward(t,i)
        # let target arm be the worst arm forever
        target = np.argmin(self.reward[t])
        # push everything to the min = 0
        attacked = np.random.normal(0, 0.01)
        cost = np.abs(true-attacked)
        if i == target:
            return true, 0
        else:
            return attacked, cost
        
    def oracle_strongest(self, t, i, eps = 0.01):
        true = self.random_reward(t,i)
        # let target arm be the worst arm forever
        target = np.argmin(self.reward[t])
        min_reward = min(self.reward[t])
        attacked = np.random.normal(min_reward-eps, 0.01)
        cost = np.abs(true-attacked)
        
        if i == target or true <= attacked:
            return true, 0
        else:
            return attacked, cost
    
    def garcelon_topN(self, t, i):
        N = int(0.5*self.K) # corrupt top half arms
        topN_index = [x[0] for x in sorted(enumerate(self.reward[t]), key=lambda x: x[1])[-N:]]
        true = self.random_reward(t,i)
        attacked = np.random.normal(0, 0.01)
        cost = np.abs(true-attacked)
        if i not in topN_index:
            return true, 0
        else:
            return attacked, cost
        
    def oracle_topN(self, t, i, eps=0.01):
        N = int(0.5*self.K) # corrupt top half arms
        topN_index = [x[0] for x in sorted(enumerate(self.reward[t]), key=lambda x: x[1])[-N:]]
        true = self.random_reward(t,i)
        min_reward = min(self.reward[t])
        attacked = np.random.normal(min_reward-eps, 0.01)
        cost = np.abs(true-attacked)
        
        if i not in topN_index or true <= attacked:
            return true, 0
        else:
            return attacked, cost
    
    def flip_theta(self, t, i):
        true = self.random_reward(t,i)
        attacked = np.random.normal(-self.reward[t][i], 0.01)
        cost = np.abs(true-attacked)
        return attacked, cost