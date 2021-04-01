#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import math

def auto_tuning(logw, p, reward, index, gamma):
    Kexp = len(logw)
    # update exp3 components
    logw[index] += (gamma/ Kexp * reward / p[index])
    # run exp3 to determine next exploration rate
    max_logw = np.max(logw)
    w = np.exp(logw - max_logw)
    p = gamma/ Kexp + (1-gamma) * w/sum(w)
    nxt_index = np.random.choice(Kexp, p=p)
    return logw, p, nxt_index


def op_tuning(s, f, reward, index):
    Kexp = len(s)
    r = np.random.binomial(1, max(0, min(reward,1)))
    s[index] += r
    f[index] += (1-r)
    beta = np.array([np.random.beta(s[i], f[i]) for i in range(Kexp)])
    index = np.argmax(beta)
    return s, f, index
'''
def op_tuning(s, f, reward, index):
    Kexp = len(s)
    # r = np.random.binomial(1, max(0, min(reward,1)))
    s[index] += reward
    f[index] += (1-reward)
    beta = np.array([np.random.beta(s[i], f[i]) for i in range(Kexp)])
    index = np.argmax(beta)
    return s, f, index
'''

def auto_tuning_exp31(logw, p, reward, index, r, ghat):
    Kexp = len(logw)
    gr = Kexp * math.log(Kexp) / (np.exp(1)-1) * 4**r
    gamma = min(1, math.sqrt(Kexp * math.log(Kexp) / (np.exp(1)-1) / gr ) )
    
    ghat[index] += reward / p[index]
    if np.max(ghat) <= gr- Kexp/gamma:
        logw, p, nxt_index = auto_tuning(logw, p, reward, index, gamma)
        return logw, p, nxt_index, r, ghat
    else:
        while True:
            r += 1
            gr = Kexp * math.log(Kexp) / (np.exp(1)-1) * 4**r
            gamma = min(1, math.sqrt(Kexp * math.log(Kexp) / (np.exp(1)-1) / gr ) )
            if np.max(ghat) <= gr- Kexp/gamma:
                break 
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        logw, p, nxt_index = auto_tuning(logw, p, reward, index, gamma)
        return logw, p, nxt_index, r, ghat # auto_tuning_exp31(logw, p, reward, index, r, ghat)
    
    
    
    
    
