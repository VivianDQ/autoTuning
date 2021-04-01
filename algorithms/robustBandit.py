import numpy as np
import random
import math

class robustbandit_reward:
    def __init__(self, attack_type, class_context, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.attack = getattr(self.data, attack_type) 
        self.budget = self.data.attack_budget
        
    def robustbandit(self, delta=0.1, lamda=0.1):
        
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)
        
        # initialization for exp3 algo
        # the possible choices for C is in J
        J = [0] + [2**x for x in range(int(math.log(T)/math.log(2))+1)]
        Kexp = len(J)
        w = np.ones(Kexp)
        gamma = min(1, math.sqrt( Kexp*math.log(Kexp) / ( (np.exp(1)-1) * T ) ) )
        
        gamma_explore = 0
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            
            # run exp3 to determine c
            p = gamma/ Kexp + (1-gamma) * w/sum(w)
            indexC = np.random.choice(Kexp, p=p)
            c = J[indexC]

            explore1 = 0.01*math.sqrt( d*math.log((t*self.data.max_norm**2/lamda+1)/delta) ) + math.sqrt(lamda)
            explore2 = math.sqrt(gamma_explore)
            
            ucb = np.zeros(K)
            width = np.zeros(K)
            for arm in range(K):
                # c is chosen by exp3
                ucb[arm] = feature[arm].dot(theta_hat)
                width[arm] = math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            ucb_idx = ucb + width * (explore1 + explore2*c)
            pull = np.argmax(ucb_idx)
            gamma_explore += width[pull]**2
            
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
            
            # update exp3 components
            w[indexC] *= np.exp(gamma/ Kexp * observe_r / p[indexC]) # (observe_r/4+0.5)
        return regret

class robustbandit_context:
    def __init__(self, attack_type, class_context, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.attack_type = attack_type
        self.attack = getattr(self.data, attack_type) 
        self.budget = self.data.attack_budget
        
    def robustbandit(self, delta=0.1, lamda=0.1):
        
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)
        
        # initialization for exp3 algo
        # the possible choices for C is in J
        J = [0] + [2**x for x in range(int(math.log(T)/math.log(2))+1)]
        Kexp = len(J)
        w = np.ones(Kexp)
        gamma = min(1, math.sqrt( Kexp*math.log(Kexp) / ( (np.exp(1)-1) * T ) ) )
        
        gamma_explore = 0
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            
            # run exp3 to determine c
            p = gamma/ Kexp + (1-gamma) * w/sum(w)
            indexC = np.random.choice(Kexp, p=p)
            c = J[indexC]
                
            explore1 = 0.01*math.sqrt( d*math.log((t*self.data.max_norm**2/lamda+1)/delta) ) + math.sqrt(lamda)
            explore2 = math.sqrt(gamma_explore)
            
            ucb = np.zeros(K)
            width = np.zeros(K)
            for arm in range(K):
                # c is chosen by exp3
                ucb[arm] = feature[arm].dot(theta_hat)
                width[arm] = math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            ucb_idx = ucb + width * (explore1 + explore2*c)
            pull = np.argmax(ucb_idx)
            
            if self.budget > 0:
                feature, cost = self.attack(t, pull)
                self.budget -= cost
                
                ucb = np.zeros(K)
                width = np.zeros(K)
                for arm in range(K):
                    # c is chosen by exp3
                    ucb[arm] = feature[arm].dot(theta_hat)
                    width[arm] = math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
                ucb_idx = ucb + width * (explore1 + explore2*c)
                pull = np.argmax(ucb_idx)
                
            gamma_explore += width[pull]**2
            observe_r = self.data.random_reward(t,pull)
            B += np.outer(feature[pull], feature[pull])
            B_inv = np.linalg.inv(B)
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            
            # update exp3 components
            w[indexC] *= np.exp(gamma/ Kexp * observe_r / p[indexC])
        return regret