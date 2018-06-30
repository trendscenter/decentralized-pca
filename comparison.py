#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:08:19 2018

@author: Harshvardhan
"""

import numpy as np
import urllib.request
from local_ancillary import base_PCA

if __name__ == '__main__':
    # URL from where the data was loaded
    # More info at https://archive.ics.uci.edu/ml/datasets/Madelon
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/'
    train_url = urllib.request.urlopen(url + 'madelon_train.data')
    test_url = urllib.request.urlopen(url + 'madelon_test.data')
    valid_url = urllib.request.urlopen(url + 'madelon_valid.data')

    # Load train, test and validation data
    print("loading train, test and validation data fom UCI")
    train = np.loadtxt(train_url)  # (2000, 500)
    test = np.loadtxt(test_url)  # (1800, 500)
    valid = np.loadtxt(valid_url)  # (600, 500)

    # Putting all the data together
    print("stacking train, test and validation data")
    all_data = np.vstack((train, test, valid))  # (4400, 500)

    # Split data into two sites
    print("splitting data into 2 parts")
    local0, local1 = np.split(all_data, 2)
    
    # concatenate sideways
    print("stacking split data sideways")
    pooled_data = np.hstack((local0, local1))

    # save each part in its correspinding local# folder
    print("saving data in the corresponding local sites folders")
    np.savetxt('test/local0/simulatorRun/local0.data', local0, fmt='%d')
    np.savetxt('test/local1/simulatorRun/local1.data', local1, fmt='%d')

    print("running base_PCA on pooled data")
    pooled_pc, b, c = base_PCA(pooled_data, num_PC=100, axis=1, whitening=True)

    print("saving pooled results to a pool_red.data")
    np.savetxt('pooled_pc.data', pooled_pc, fmt='%.6f')

    # Assuming PCA was already ran in the coinstac simulator
    print("loading decentralized results data")
    decent_pc = np.loadtxt('test/output/remote/simulatorRun/decent_pc.data')

    if np.array_equal(pooled_pc, decent_pc):
        print("results from pooled and decentralized match")
    else:
        print("results from pooled and decentralized don't match")
    
