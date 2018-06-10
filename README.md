# decentralized-pca
This repository contains the decentralized PCA code written for the new coinstac simulator (v3.1.6). It contains the following files:
1. local.py - for computing the local PCA on local data and sending the partial square root to the master.
2. master.py - for aggregation of the partial square roots sent by local sites and releasing the energy captured in the top-K subspace
3. compspec.json - computation specifications
4. generate_data.py - python file for generating data, not used in the simulator
