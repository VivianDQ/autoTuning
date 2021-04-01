import numpy as np
import random
import math

class Greedy_reward:
    def __init__(self, attack_type, class_context, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.attack = getattr(self.data, attack_type) 
        self.budget = self.data.attack_budget
        
    def greedy(self, lamda=0.1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)
        
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat)
            pull = np.argmax(ucb_idx)
            
            if self.budget < 0:
                observe_r = self.data.random_reward(t,pull)
            else:
                observe_r, cost = self.attack(t, pull)
                self.budget -= cost
            B += np.outer(feature[pull], feature[pull])
            B_inv = np.linalg.inv(B)
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)

            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret

class Greedy_context:
    def __init__(self, attack_type, class_context, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.attack = getattr(self.data, attack_type) 
        self.budget = self.data.attack_budget
        
    def greedy(self, lamda=0.1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)
        
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat)
            pull = np.argmax(ucb_idx)
            
            if self.budget > 0:
                feature, cost = self.attack(t, pull)
                self.budget -= cost
                
                ucb_idx = [0]*K
                for arm in range(K):
                    ucb_idx[arm] = feature[arm].dot(theta_hat)
                pull = np.argmax(ucb_idx)
            observe_r = self.data.random_reward(t,pull)
            B += np.outer(feature[pull], feature[pull])
            B_inv = np.linalg.inv(B)
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)

            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret