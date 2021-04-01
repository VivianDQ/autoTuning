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