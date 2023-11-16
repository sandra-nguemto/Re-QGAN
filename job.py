from qGAN import *
import numpy as np
import pickle 
from sklearn.decomposition import PCA
import argparse
import datetime
import os
import sys
import time

# make the number of qubits a command line argument
parser = argparse.ArgumentParser()
parser.add_argument('--nqubits', type=int, default=4)
parser.add_argument('--nshots', type=int, default=1024)
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--nsteps', type=int, default=1000)
parser.add_argument('--nsamples', type=int, default=50)
parser.add_argument('--digit', type=int, default=0)
parser.add_argument('--backend', type=str, default='simulator')
#
args = parser.parse_args()
nqubits = args.nqubits
nshots = args.nshots
nepochs = args.nepochs
nsteps = args.nsteps
nsamples = args.nsamples
digit = args.digit
backend = args.backend
#
# today's date 
date = datetime.datetime.today().strftime('%Y-%m-%d')
# compression of the data
data, data_decomp, pca_ = reduced_data(n_q=nqubits, 
                                       shape='digits', 
                                       n_comp_max=nsamples, 
                                       idx=digit)
# configure the class object
algo = qgan(qubits = [i for i in range(nqubits)], 
            data = data, 
            backend = backend, 
            qc_real_build=False, 
            mock_backend=False, 
            opt_maxevals=nsteps)

# run the algorithm
output = algo.game(epochs=nepochs)
# store the results
pickle.dump(output, open('results/reqgan_{}_{}_{}_{}_{}_{}.pkl'.format(nqubits, nsamples, digit, nepochs, date, backend), 'wb'))
# Example of how to execute the algorithm in the command line
# python job.py --nqubits 4 --nepochs 100 --nsteps 1000 --nsamples 25 --digit 0 --backend qasm_simulator
