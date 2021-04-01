import numpy as np
import random
import math

class LinUCB_oracle_reward:
    def __init__(self, attack_type, class_context, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.attack = getattr(self.data, attack_type) 
        self.budget = self.data.attack_budget
        
    def linucb_oracle(self, delta=0.1, lamda=0.1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)
        current_cost = 0
        
        t = 0
        gamma_explore = 0
        while t<T and self.budget>0:
            feature = self.data.fv[t]
            K = len(feature)

            explore1 = 0.01*math.sqrt( d*math.log((t*self.data.max_norm**2/lamda+1)/delta) ) + math.sqrt(lamda)
            explore2 = math.sqrt(gamma_explore)
            ucb = np.zeros(K)
            width = np.zeros(K)
            for arm in range(K):
                ucb[arm] = feature[arm].dot(theta_hat)
                width[arm] = math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            ucb_idx = ucb + (explore1 + explore2*current_cost) * width
            pull = np.argmax(ucb_idx)
            gamma_explore += width[pull]**2 
            
            if self.budget < 0:
                observe_r = self.data.random_reward(t,pull)
            else:
                observe_r, cost = self.attack(t, pull)
                self.budget -= cost
                current_cost += cost
            B += np.outer(feature[pull], feature[pull])
            B_inv = np.linalg.inv(B)
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            t += 1
            
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)
        
        for ite in range(t, T):
            feature = self.data.fv[ite]
            K = len(feature)

            explore = 0.01*math.sqrt( d*math.log(((ite-t+1)*self.data.max_norm**2/lamda+1)/delta) ) + math.sqrt(lamda)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat) + explore * math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_reward(ite,pull)
            B += np.outer(feature[pull], feature[pull])
            B_inv = np.linalg.inv(B)
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            regret[ite] = regret[ite-1] + self.data.optimal[ite] - self.data.reward[ite][pull]
        return regret

class LinUCB_oracle_context:
    def __init__(self, attack_type, class_context, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.attack = getattr(self.data, attack_type) 
        self.budget = self.data.attack_budget
        
    def linucb_oracle(self, delta=0.1, lamda=0.1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)
        current_cost = 0
        
        t = 0
        gamma_explore = 0
        while t<T and self.budget>0:
            feature = self.data.fv[t]
            K = len(feature)
            
            explore1 = 0.01*math.sqrt( d*math.log((t*self.data.max_norm**2/lamda+1)/delta) ) + math.sqrt(lamda)
            explore2 = math.sqrt(gamma_explore)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat) + (explore1+explore2*current_cost) * math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            pull = np.argmax(ucb_idx)

            feature, cost = self.attack(t, pull)
            self.budget -= cost

            ucb = np.zeros(K)
            width = np.zeros(K)
            for arm in range(K):
                ucb[arm] = feature[arm].dot(theta_hat)
                width[arm] = math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            ucb_idx = ucb + (explore1+explore2*current_cost) * width
            pull = np.argmax(ucb_idx)
            gamma_explore += width[pull]**2 

            observe_r = self.data.random_reward(t,pull)
            B += np.outer(feature[pull], feature[pull])
            B_inv = np.linalg.inv(B)
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            t += 1
            current_cost += cost
            
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)
        
        for ite in range(t, T):
            feature = self.data.fv[ite]
            K = len(feature)

            explore = 0.01*math.sqrt( d*math.log(((ite-t+1)*self.data.max_norm**2/lamda+1)/delta) ) + math.sqrt(lamda)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat) + explore * math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_reward(ite,pull)
            B += np.outer(feature[pull], feature[pull])
            B_inv = np.linalg.inv(B)
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            regret[ite] = regret[ite-1] + self.data.optimal[ite] - self.data.reward[ite][pull]
        return regret