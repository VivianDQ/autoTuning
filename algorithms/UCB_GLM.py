import numpy as np
import random
import math
from sklearn.linear_model import LogisticRegression
from algorithms.AutoTuning import *

class UCB_GLM:
    def __init__(self, class_context, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d

    def glmucb_theoretical_explore(self, lamda=1, delta=0.1, explore = -1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        B = np.identity(d) * lamda
        theta_hat = np.zeros(d)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        theta = np.zeros(d)
        for t in range(2):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull) 
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            B += np.outer(feature[pull], feature[pull])
        if y[0] == y[1]:
            y[1] = 1-y[0]
        # random pull in the first two rounds to make sure y[0] != y[1]
        B_inv = np.linalg.inv(B)
        for t in range(2, T):
            # when explore = -1, which is impossible, use theoretical value
            # otherwise, it means I have specify a fixed value of explore in the code
            # specify a fixed value for explore is only for grid serach
            if explore == -1:
                explore = self.data.sigma*math.sqrt( d*math.log((t*self.data.max_norm**2/lamda+1)/delta) ) + math.sqrt(lamda)
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            clf = LogisticRegression(penalty = 'l2', C = lamda, fit_intercept = False, solver = 'lbfgs').fit(X, y)
            theta = clf.coef_[0]
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            tmp = B_inv.dot(feature[pull])
            B += np.outer(feature[pull], feature[pull])
            B_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret
    
    def glmucb_auto(self, explore_rates, lamda=1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        B = np.identity(d) * lamda
        theta_hat = np.zeros(d)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        theta = np.zeros(d)
        for t in range(2):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull) 
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            B += np.outer(feature[pull], feature[pull])
        if y[0] == y[1]:
            y[1] = 1-y[0]
        # random pull in the first two rounds to make sure y[0] != y[1]
        B_inv = np.linalg.inv(B)
        
        # initialization for exp3 algo
        Kexp = len(explore_rates)
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        gamma = min(1, math.sqrt( Kexp*math.log(Kexp) / ( (np.exp(1)-1) * T ) ) )
        # random initial explore rate
        index = np.random.choice(Kexp)
        explore = explore_rates[index]
        for t in range(2, T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            clf = LogisticRegression(penalty = 'l2', C = lamda, fit_intercept = False, solver = 'lbfgs').fit(X, y)
            theta = clf.coef_[0]
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            tmp = B_inv.dot(feature[pull])
            B += np.outer(feature[pull], feature[pull])
            B_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            
            # update explore rates by auto_tuning
            logw, p, index = auto_tuning(logw, p, observe_r, index, gamma)
            explore = explore_rates[index]
        return regret    
    
    def glmucb_op(self, explore_rates, lamda=1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        B = np.identity(d) * lamda
        theta_hat = np.zeros(d)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        theta = np.zeros(d)
        for t in range(2):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull) 
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            B += np.outer(feature[pull], feature[pull])
        if y[0] == y[1]:
            y[1] = 1-y[0]
        # random pull in the first two rounds to make sure y[0] != y[1]
        B_inv = np.linalg.inv(B)
        
        # initialization for op_tuning
        Kexp = len(explore_rates)
        s = np.ones(Kexp)
        f = np.ones(Kexp)
        index = np.random.choice(Kexp)
        explore = explore_rates[index]
        for t in range(2, T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            clf = LogisticRegression(penalty = 'l2', C = lamda, fit_intercept = False, solver = 'lbfgs').fit(X, y)
            theta = clf.coef_[0]
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            tmp = B_inv.dot(feature[pull])
            B += np.outer(feature[pull], feature[pull])
            B_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            
            # update explore rates by op_tuning
            s, f, index = op_tuning(s, f, observe_r, index)
            explore = explore_rates[index]
        return regret
    
    def glmucb_auto_3layer(self, explore_rates, lamdas):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        theta_hat = np.zeros(d)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        theta = np.zeros(d)
        
        xxt = np.zeros((d,d))
        for t in range(2):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull) 
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            xxt += np.outer(feature[pull], feature[pull])
        if y[0] == y[1]:
            y[1] = 1-y[0]
        # random pull in the first two rounds to make sure y[0] != y[1]
        
        # initialization for exp3 algo
        Klam = len(lamdas)
        loglamw = np.zeros(Klam)
        plam = np.ones(Klam) / Klam
        gamma_lam = min(1, math.sqrt( Klam*math.log(Klam) / ( (np.exp(1)-1) * T ) ) )
        # random initial explore rate
        index_lam = np.random.choice(Klam)
        lamda = lamdas[index_lam]

        B_inv = np.linalg.inv(xxt + lamda*np.identity(d))
        for t in range(2, T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            clf = LogisticRegression(penalty = 'l2', C = lamda, fit_intercept = False, solver = 'lbfgs').fit(X, y)
            theta = clf.coef_[0]
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)

            # update explore rates by auto_tuning
            logw, p, index = auto_tuning(logw, p, observe_r, index, gamma)
            explore = explore_rates[index]
            loglamw, plam, index_lam = auto_tuning(loglamw, plam, observe_r, index_lam, gamma_lam)
            lamda = lamdas[index_lam]
            
            xxt += np.outer(feature[pull], feature[pull])
            B_inv = np.linalg.inv(xxt + lamda*np.identity(d))
            
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret 
    
    def glmucb_auto_combined(self, explore_rates, lamdas):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        theta_hat = np.zeros(d)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        theta = np.zeros(d)
        
        xxt = np.zeros((d,d))
        for t in range(2):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull) 
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            xxt += np.outer(feature[pull], feature[pull])
        if y[0] == y[1]:
            y[1] = 1-y[0]
        # random pull in the first two rounds to make sure y[0] != y[1]
        
        # initialization for exp3 algo
        explore_lamda = np.array(np.meshgrid(explore_rates, lamdas)).T.reshape(-1,2)
        Kexp = len(explore_lamda)
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        gamma = min(1, math.sqrt( Kexp*math.log(Kexp) / ( (np.exp(1)-1) * T ) ) )
        # random initial explore rate
        index = np.random.choice(Kexp)
        explore, lamda = explore_lamda[index]

        B_inv = np.linalg.inv(xxt + lamda*np.identity(d))
        for t in range(2, T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            clf = LogisticRegression(penalty = 'l2', C = lamda, fit_intercept = False, solver = 'lbfgs').fit(X, y)
            theta = clf.coef_[0]
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)

            # update explore rates by auto_tuning
            logw, p, index = auto_tuning(logw, p, observe_r, index, gamma)
            explore, lamda = explore_lamda[index]
            
            xxt += np.outer(feature[pull], feature[pull])
            B_inv = np.linalg.inv(xxt + lamda*np.identity(d))
            
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret 