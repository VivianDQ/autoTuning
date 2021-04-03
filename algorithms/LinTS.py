import numpy as np
import random
import math
from algorithms.AutoTuning import *

class LinTS:
    def __init__(self, class_context, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        
    def lints_theoretical_explore(self, lamda=1, delta=0.1):
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
            
            explore = self.data.sigma*math.sqrt( d*math.log((t*self.data.max_norm**2/lamda+1)/delta) ) + math.sqrt(lamda)*self.data.max_norm
            theta_ts = np.random.multivariate_normal(theta_hat, explore**2*B_inv)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_ts)
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t,pull)
            
            B += np.outer(feature[pull], feature[pull])
            tmp = B_inv.dot(feature[pull])
            B_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)

            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret
    
    def lints_auto(self, explore_rates, lamda=1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)
        
        # initialization for exp3 algo
        # the following two lines are an ideal set of "explore_rates"
        # min_rate, max_rate = 0, 2 * (int(math.sqrt(d * math.log(T**2+T) ) + math.sqrt(lamda)) + 1)
        # J = np.arange(min_rate, max_rate, explore_interval_length)
        Kexp = len(explore_rates)
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        gamma = min(1, math.sqrt( Kexp*math.log(Kexp) / ( (np.exp(1)-1) * T ) ) )
        # random initial explore rate
        index = np.random.choice(Kexp)
        explore = explore_rates[index]
        
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            
            theta_ts = np.random.multivariate_normal(theta_hat, explore**2*B_inv)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_ts)
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t,pull)
            
            # update linucb
            tmp = B_inv.dot(feature[pull])
            B_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            
            # update explore rates by auto_tuning
            logw, p, index = auto_tuning(logw, p, observe_r, index, gamma)
            explore = explore_rates[index]
        return regret
    
    def lints_op(self, explore_rates, lamda=1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)
        
        # initialization for exp3 algo
        # the possible choices for C is in J
        # the following two lines are an ideal set of "explore_rates"
        # min_rate, max_rate = 0, 2 * (int(math.sqrt(d * math.log(T**2+T) ) + math.sqrt(lamda)) + 1)
        # J = np.arange(min_rate, max_rate, explore_interval_length)
        Kexp = len(explore_rates)
        s = np.ones(Kexp)
        f = np.ones(Kexp)
        index = np.random.choice(Kexp)
        explore = explore_rates[index]
        
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            
            
            theta_ts = np.random.multivariate_normal(theta_hat, explore**2*B_inv)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_ts)
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t,pull)
            
            # update linucb
            tmp = B_inv.dot(feature[pull])
            B_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            
            # update explore rates by auto_tuning
            s, f, index = op_tuning(s, f, observe_r, index)
            explore = explore_rates[index]
        return regret
    
    def lints_auto_3layer(self, explore_rates, lamdas):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        theta_hat = np.zeros(d)
        
        # initialization for exp3 algo
        # the possible choices for C is in J
        # the following two lines are an ideal set of "explore_rates"
        # min_rate, max_rate = 0, 2 * (int(math.sqrt(d * math.log(T**2+T) ) + math.sqrt(lamda)) + 1)
        # J = np.arange(min_rate, max_rate, explore_interval_length)
        Kexp = len(explore_rates)
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        gamma = min(1, math.sqrt( Kexp*math.log(Kexp) / ( (np.exp(1)-1) * T ) ) )
        # random initial explore rate
        index = np.random.choice(Kexp)
        explore = explore_rates[index]
        
        Klam = len(lamdas)
        loglamw = np.zeros(Klam)
        plam = np.ones(Klam) / Klam
        gamma_lam = min(1, math.sqrt( Klam*math.log(Klam) / ( (np.exp(1)-1) * T ) ) )
        # random initial explore rate
        index_lam = np.random.choice(Klam)
        lamda = lamdas[index_lam]
        
        xxt = np.zeros((d,d))
        B_inv = np.identity(d) / lamda
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            
            theta_ts = np.random.multivariate_normal(theta_hat, explore**2*B_inv)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_ts)
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t,pull)
            
            # update linucb
            xxt += np.outer(feature[pull], feature[pull])
            B_inv = np.linalg.inv(xxt + lamda*np.identity(d))
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            
            # update explore rates by auto_tuning
            logw, p, index = auto_tuning(logw, p, observe_r, index, gamma)
            explore = explore_rates[index]
            loglamw, plam, index_lam = auto_tuning(loglamw, plam, observe_r, index_lam, gamma_lam)
            lamda = lamdas[index_lam]
        return regret