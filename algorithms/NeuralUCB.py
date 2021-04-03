import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils import data

import numpy as np
import random
import math
import matplotlib.pyplot as plot
import collections
import time
from algorithms.AutoTuning import *



class Net(nn.Module):
    def __init__(self, m):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(m, m, bias = False)
        self.fc2 = nn.Linear(m, 1, bias = False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def grad(self, x, w1, w2, m):
        x = x.reshape((1,len(x)))
        x_fc1 = x.dot(w1)
        x_relu = np.maximum(x_fc1, 0)
        x_fc2 = x_relu.dot(w2)
        y_pred = math.sqrt(m) * x_fc2

        grad_x_fc2 = math.sqrt(m) * np.ones(x_fc2.shape)
        grad_w2 = x_relu * grad_x_fc2
        grad_x_relu = grad_x_fc2 * w2.T
        grad_x_fc1 = grad_x_relu.copy()
        grad_x_fc1[x_fc1 < 0] = 0
        grad_w1 = x.T.dot(grad_x_fc1)
        return np.concatenate([grad_w1.flatten(), grad_w2.flatten()])

class NeuralUCB:
    def __init__(self, class_context, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.random_sample = self.data.random_sample
        
    def train(self, X, Y, W, Wl, max_ite, m, optimize = 'sgd', eta = 0.1): # or optimize = 'adagrad'
        tensor_x = torch.Tensor(X) # transform to torch tensor
        tensor_y = torch.Tensor(Y)
        my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y) # create your datset
        my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size = 50) 
        net = Net(m)
        
        if torch.cuda.is_available():
            dev = "cuda:0" 
        else:
            dev = "cpu"
        device = torch.device(dev)    
        net = net.to(device)
        
        net.fc1.weight = torch.nn.Parameter(W)
        net.fc2.weight = torch.nn.Parameter(Wl)
        criterion = nn.MSELoss()
        if optimize == 'adagrad':
            optimizer = optim.Adagrad(net.parameters(), lr=eta, weight_decay=0)
        else:
            optimizer = optim.SGD(net.parameters(), lr=eta, weight_decay = 0)
        for epoch in range(max_ite):
            running_loss = 0.0
            for i, data in enumerate(my_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                x, y = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net.forward(x)
                loss = criterion(math.sqrt(m)*outputs, y) + m/2 * (net.fc1.weight - W).norm(2)**2 + m/2*(net.fc2.weight - Wl).norm(2)**2
                loss.backward()
                optimizer.step()
        return net
        
    def neuralucb_theoretical_explore(self, gamma_t = 1, eta = 0.1, m = 20, lamda = 1, optimize = 'sgd'): # or optimize = 'adagrad'
        T = self.T
        K = self.data.K
        d = self.data.d
        regret = np.zeros(self.T)
        
        net = Net(m)
        w = np.random.normal(0,4/m,(m//2,m//2))
        W = np.block([
        [w,               np.zeros((m//2, m//2))],
        [np.zeros((m//2, m//2)), w              ]
        ])
        w_l = list( np.random.normal(0,2/m,m//2) )
        W_L = w_l + [-x for x in w_l]
        tensor_W = torch.Tensor(W)
        tensor_Wl = torch.Tensor(W_L)

        X = []
        Y = []
        Zt = np.identity(m*m + m)*lamda
        Zt_inv = np.linalg.inv(Zt)
        
        ucb_idx = np.zeros(K)
        grad = [np.zeros(d)] * K
    
        for t in range(T):
            feature = self.data.fv[t]
            w1 = np.array(net.state_dict()['fc1.weight'])
            w2 = np.array(net.state_dict()['fc2.weight']).reshape((m,1))
            for arm in range(K):
                grad[arm] = net.grad(feature[arm], w1, w2, m)
                ucb_idx[arm] = net.forward(torch.Tensor(feature[arm])) + gamma_t / math.sqrt(m) * math.sqrt(grad[arm].dot(Zt_inv).dot(grad[arm]))
            
            pull = np.argmax(ucb_idx)
            observe_r = self.random_sample(t, pull)
            Y.append(observe_r)
            X.append(feature[pull])
            if t%50 == 49:
                net = self.train(X, Y, tensor_W, tensor_Wl, 50, m, optimize = optimize, eta = eta)
                w1 = np.array(net.state_dict()['fc1.weight'])
                w2 = np.array(net.state_dict()['fc2.weight']).reshape((m,1))
                grad[pull] = net.grad(feature[arm], w1, w2, m)
            left = Zt_inv.dot(grad[pull])
            Zt_inv -= np.outer(left, left) / (1 + grad[pull].dot(Zt_inv).dot(grad[pull]))
            
            
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
           
        return regret
    
    def neuralucb_auto(self, explore_rates, eta = 0.1, m = 20, lamda = 1, optimize = 'sgd'): # or optimize = 'adagrad'
        T = self.T
        K = self.data.K
        d = self.data.d
        regret = np.zeros(self.T)
        
        net = Net(m)
        w = np.random.normal(0,4/m,(m//2,m//2))
        W = np.block([
        [w,               np.zeros((m//2, m//2))],
        [np.zeros((m//2, m//2)), w              ]
        ])
        w_l = list( np.random.normal(0,2/m,m//2) )
        W_L = w_l + [-x for x in w_l]
        tensor_W = torch.Tensor(W)
        tensor_Wl = torch.Tensor(W_L)

        X = []
        Y = []
        Zt = np.identity(m*m + m)*lamda
        Zt_inv = np.linalg.inv(Zt)
        
        ucb_idx = np.zeros(K)
        grad = [np.zeros(d)] * K
        
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
        
        for t in range(T):
            feature = self.data.fv[t]
            w1 = np.array(net.state_dict()['fc1.weight'])
            w2 = np.array(net.state_dict()['fc2.weight']).reshape((m,1))
            for arm in range(K):
                grad[arm] = net.grad(feature[arm], w1, w2, m)
                ucb_idx[arm] = net.forward(torch.Tensor(feature[arm])) + explore / math.sqrt(m) * math.sqrt(grad[arm].dot(Zt_inv).dot(grad[arm]))
            
            pull = np.argmax(ucb_idx)
            observe_r = self.random_sample(t, pull)
            Y.append(observe_r)
            X.append(feature[pull])
            if t%50 == 49:
                net = self.train(X, Y, tensor_W, tensor_Wl, 50, m, optimize = optimize, eta = eta)
                w1 = np.array(net.state_dict()['fc1.weight'])
                w2 = np.array(net.state_dict()['fc2.weight']).reshape((m,1))
                grad[pull] = net.grad(feature[arm], w1, w2, m)
            left = Zt_inv.dot(grad[pull])
            Zt_inv -= np.outer(left, left) / (1 + grad[pull].dot(Zt_inv).dot(grad[pull]))
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            
            # update explore rates by auto_tuning
            logw, p, index = auto_tuning(logw, p, observe_r, index, gamma)
            explore = explore_rates[index]
        return regret
    
    def neuralucb_auto_3layer(self, explore_rates, etas, m = 20, lamda = 1, optimize = 'sgd'): # or optimize = 'adagrad'
        T = self.T
        K = self.data.K
        d = self.data.d
        regret = np.zeros(self.T)
        
        net = Net(m)
        w = np.random.normal(0,4/m,(m//2,m//2))
        W = np.block([
        [w,               np.zeros((m//2, m//2))],
        [np.zeros((m//2, m//2)), w              ]
        ])
        w_l = list( np.random.normal(0,2/m,m//2) )
        W_L = w_l + [-x for x in w_l]
        tensor_W = torch.Tensor(W)
        tensor_Wl = torch.Tensor(W_L)

        X = []
        Y = []
        Zt = np.identity(m*m + m)*lamda
        Zt_inv = np.linalg.inv(Zt)
        
        ucb_idx = np.zeros(K)
        grad = [np.zeros(d)] * K
        
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
        
        Klam = len(etas)
        loglamw = np.zeros(Klam)
        plam = np.ones(Klam) / Klam
        gamma_lam = min(1, math.sqrt( Klam*math.log(Klam) / ( (np.exp(1)-1) * T ) ) )
        # random initial explore rate
        index_lam = np.random.choice(Klam)
        eta = etas[index_lam]
        
        for t in range(T):
            feature = self.data.fv[t]
            w1 = np.array(net.state_dict()['fc1.weight'])
            w2 = np.array(net.state_dict()['fc2.weight']).reshape((m,1))
            for arm in range(K):
                grad[arm] = net.grad(feature[arm], w1, w2, m)
                ucb_idx[arm] = net.forward(torch.Tensor(feature[arm])) + explore / math.sqrt(m) * math.sqrt(grad[arm].dot(Zt_inv).dot(grad[arm]))
            
            pull = np.argmax(ucb_idx)
            observe_r = self.random_sample(t, pull)
            Y.append(observe_r)
            X.append(feature[pull])
            if t%50 == 49:
                net = self.train(X, Y, tensor_W, tensor_Wl, 50, m, optimize = optimize, eta = eta)
                w1 = np.array(net.state_dict()['fc1.weight'])
                w2 = np.array(net.state_dict()['fc2.weight']).reshape((m,1))
                grad[pull] = net.grad(feature[arm], w1, w2, m)
            left = Zt_inv.dot(grad[pull])
            Zt_inv -= np.outer(left, left) / (1 + grad[pull].dot(Zt_inv).dot(grad[pull]))
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            
            # update explore rates by auto_tuning
            logw, p, index = auto_tuning(logw, p, observe_r, index, gamma)
            explore = explore_rates[index]
            
            loglamw, plam, index_lam = auto_tuning(loglamw, plam, observe_r, index_lam, gamma_lam)
            eta = etas[index_lam]
        return regret
    
    def neuralucb_op(self, explore_rates, eta = 0.1, m = 20, lamda = 1, optimize = 'sgd'): # or optimize = 'adagrad'
        T = self.T
        K = self.data.K
        d = self.data.d
        regret = np.zeros(self.T)
        
        net = Net(m)
        w = np.random.normal(0,4/m,(m//2,m//2))
        W = np.block([
        [w,               np.zeros((m//2, m//2))],
        [np.zeros((m//2, m//2)), w              ]
        ])
        w_l = list( np.random.normal(0,2/m,m//2) )
        W_L = w_l + [-x for x in w_l]
        tensor_W = torch.Tensor(W)
        tensor_Wl = torch.Tensor(W_L)

        X = []
        Y = []
        Zt = np.identity(m*m + m)*lamda
        Zt_inv = np.linalg.inv(Zt)
        
        ucb_idx = np.zeros(K)
        grad = [np.zeros(d)] * K
        
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
            w1 = np.array(net.state_dict()['fc1.weight'])
            w2 = np.array(net.state_dict()['fc2.weight']).reshape((m,1))
            for arm in range(K):
                grad[arm] = net.grad(feature[arm], w1, w2, m)
                ucb_idx[arm] = net.forward(torch.Tensor(feature[arm])) + explore / math.sqrt(m) * math.sqrt(grad[arm].dot(Zt_inv).dot(grad[arm]))
            
            pull = np.argmax(ucb_idx)
            observe_r = self.random_sample(t, pull)
            Y.append(observe_r)
            X.append(feature[pull])
            if t%50 == 49:
                net = self.train(X, Y, tensor_W, tensor_Wl, 50, m, optimize = optimize, eta = eta)
                w1 = np.array(net.state_dict()['fc1.weight'])
                w2 = np.array(net.state_dict()['fc2.weight']).reshape((m,1))
                grad[pull] = net.grad(feature[arm], w1, w2, m)
            left = Zt_inv.dot(grad[pull])
            Zt_inv -= np.outer(left, left) / (1 + grad[pull].dot(Zt_inv).dot(grad[pull]))
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            
            # update explore rates by auto_tuning
            s, f, index = op_tuning(s, f, observe_r, index)
            explore = explore_rates[index]
        return regret