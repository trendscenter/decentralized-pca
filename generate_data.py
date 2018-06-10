#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 08:56:06 2018

@author: hafizimtiaz
"""

import numpy as np

D = 100
K = 20
Ns = 1000
S = 3

tmp = np.linspace(D, 1, K)
tmp = np.concatenate((tmp, 0.01 * np.random.random(D-K)))
cov = np.diag(tmp)
mu = np.zeros([D, ])

for s in range(S):
    Xs = np.random.multivariate_normal(mu, cov, Ns).T
    filename = 'value' + str(s) + '.npz'
    np.savez(filename, Xs, mu, cov, K)
    

# for multiple different subjects in each sites
num_sub_per_site = 4
for s in range(S):
    Xs = {}
    Xs['data'] = {}
    for subj in range(num_sub_per_site):
        tmp = np.random.multivariate_normal(mu, cov, np.int(Ns / num_sub_per_site)).T
        Xs['data'][subj] = tmp
    Xs['mu'] = mu
    Xs['cov'] = cov
    filename = 'value' + str(s) + '.npy'
    np.save(filename, Xs)
