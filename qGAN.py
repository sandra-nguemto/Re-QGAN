"""
The qgan class implements the entire Re-QGAN, with all its components.
(including the data encoding and decoding procedures) and trains it using CMA-ES or Adam.  
"""


from cgitb import reset
from itertools import count
from re import A
from unittest.mock import mock_open
import matplotlib.pyplot as plt
import numpy as np 
import scipy 
import cma
from PQC import *
from sklearn.decomposition import PCA
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.operators.symplectic import Pauli
import qiskit.providers.fake_provider as fake_provider
import pandas as pd
import climin
from joblib import Parallel, delayed
from tensorflow.keras.datasets import mnist
import qiskit_aer.noise as noise



def reduced_data(n_q, shape, n_comp_max = 10, idx = 0):
    """
    Function that returns reduced images, using PCA, based on the number 
    of qubits used to encode the data and build the generator.

    Inputs:

    n_q (int): number of qubits.
    shape (str): what shape are  the images we want to use.
    n_comp_max (int):  number of components from the real data set to consider for training.
    idx (int): digit we want to use images of.

    Outputs:
    
    data_red (np array)(shape: n_comp_max x number of principal components for pca):
    array of pca-reduced images.

    data_decomp (np array)(shape: n_comp_max x dimension of original image ):
    array of images reconstructed from pca transformation.

    pca_ (sklearn.decomposition._pca.PCA): the whole PCA transformation

    """
    dim = 2**n_q - 1 # number of principal components for pca, if it is less than number of training examples
    if dim > n_comp_max:
        pca_ = PCA(n_components=n_comp_max)
    else: 
        pca_ = PCA(n_components=dim)
    data = []
    if shape == 'digits':
        (image, indices), _ = mnist.load_data()
        # only pick images of a certain digit given by idx
        indices = np.where(indices == idx)
        # pick only n_comp_max images
        indices = indices[0][:n_comp_max]
        # pick the images corresponding to the indices
        image_subset = image[indices]
        image_subset = image_subset.reshape(n_comp_max, 784)
        image_subset = image_subset/255
        mnist_reduced = pca_.fit_transform(image_subset)
        mnist_reduced = np.array(mnist_reduced)
        mnist_decomp = pca_.inverse_transform(mnist_reduced)
        return mnist_reduced, mnist_decomp, pca_ 
    else:
        for i in range(n_comp_max):
            m = plt.imread(shape+'/t'+str(i)+'.jpg')
            m = to_bw(m)
            data.append(m.reshape(784))
        data = data
        data = np.array(data)/255
        data_red = pca_.fit_transform(data)
        data_decomp = pca_.inverse_transform(data_red)
        return data_red, data_decomp, pca_



def to_bw(m):
    """
    Function that changes an image into blakc and white

    Inputs:

    m (array): image in array format

    Outputs:
    
    m2 (array): black and white image in array format

    """
    
    if m[0][0].shape == (3,):
        m2 = []
        for i in range(28):
            temp = []
            for j in range(28):
                temp.append(0.299*m[i][j][0] + 0.587*m[i][j][0] + 0.114*m[i][j][0])
            temp = np.array(temp)
            m2.append(temp)
        m2 = np.array(m2)
        return m2
    else:
        return m


class qgan:
    """
    Class that implements the re-qgan.

    """
    def __init__(self, qubits, data, backend = 'simulator', qc_real_build = True, mock_backend = True, shots= 5000, opt_maxevals = 500, opt = 'cma', noise_simulation = False):
        self.opt = opt
        self.opt_maxevals = opt_maxevals
        self.qc_real_build = qc_real_build
        self.mock_backend = mock_backend
        self.qubits =  qubits 
        self.shots = shots
        n_q = len(qubits)
        self.n_q = n_q
        self.data = data 
        self.noise_simulation = noise_simulation
        #backend (which quantum simulator to use)
        ## make a dictionary of backends to avoid if statements
        backends = dict();
        backends['simulator'] = Aer.get_backend('aer_simulator')
        backends['gpu_simulator'] = Aer.get_backend('aer_simulator')
        backends['fake_bogota'] = fake_provider.FakeBogota()
        self.backend = backends[backend]
        if backend == 'gpu_simulator':
            self.backend.set_options(device='GPU')
        #
        #building the generator's pqc
        self.gen = pqc_general(qubits)
        self.qc_gen, self.p_gen = self.gen.build_qpc()
        #building the real data encoding pqc
        self.re = pqc_general(qubits)
        self.qc_re, self.p_re = self.re.build_qpc()
        #disc
        #self.qc_disc, self.p_disc, self.num_params = build_alltoall_qpc(self.n_q + 1)

        #Building the pqc for the discriminator 
        qubits_disc = qubits
        qubits_disc.append(n_q)
        qubits_disc = qubits_disc[::-1]
        self.disc = pqc_general(qubits_disc)
        self.qc_disc, self.p_disc = self.disc.build_qpc(st_to_st=True)
        self.num_params = len(self.p_disc)
        
        #
        self.res_real = []
        self.res_gen = []
        self.x_real = [] #real data pqc's parameters
        self.over_dim = False
        for dt in self.data:
            dim = 2**n_q   # Shouldn't this be 2^nq - 1?
            x = self.estereo(dt)
            if x.size < dim:  #shouldn't this be > instead of < ?
                self.over_dim = True   
                self.dim_input = x.size
                x_real = np.zeros(dim)
                for idx in range(x.size):
                    x_real[idx] = x[idx]
                self.x_real.append(x_real)
            else:
                self.x_real.append(x)
        self.iter = 0 #number of iterations
        #noise simulation
        self.p1 = 0
        self.p2 = 0 
        self.u = 0
        self.v = 0

    
    def noise_model(self,p1, p2, u, v):
        # 0 given 1: v
        # 1 given 0: u
        a = [[1-u,u],[v,1-v]]
        readouterror = noise.ReadoutError(a)
        error_1 = noise.depolarizing_error(p1, 1)
        error_2 = noise.depolarizing_error(p2, 2)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'id', 'unitary'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])  
        noise_model.add_readout_error(readouterror,[0])
        return noise_model
    
    #amplitude encoding for n-qubit quantum state with real amplitudes 
    def amp_enc(self,x):
        """
        Function that builds an re-pqc that prepares a quantum state with real amplitudes.

        Input(s): 

        x: array of real amplitudes
        n_q: number of qubits

        Output(s):

        qc: real amplitude pqc.
        """
        x = np.array(x)
        if self.n_q == 2:
            assert(x.size == 4)
            x0 = np.array(x)/np.linalg.norm(x)
            #amplitudes
            re = pqc_general(self.n_q)
            qc_par, p = re.build_qpc()
            p0 = 2*np.arccos(np.sqrt(x[0]**2 + x[1]**2))
            p1 = np.arctan(x[1]/x[0]) + np.arctan(x[2]/x[3]) 
            p2 = np.arctan(x[1]/x[0]) - np.arctan(x[2]/x[3]) 
            qc =  qc_par.bind_parameters({p:[p0,p1,p2]})
        if self.n_q == 3:
            assert(x.size == 8)
            a = np.sin(np.arctan(x[3]/x[2]))/np.cos(np.arctan(x[1]/x[0]))
            b = np.cos(np.arctan(x[5]/x[4])) / np.cos(np.arctan(x[7]/x[6]))
            c = np.sin(np.arctan(x[4]/(x[6]*b))) / np.sin(np.arctan(x[3]/(x[0]*a)))
            d = np.sin(np.arctan(x[7]/x[6])) / np.cos(np.arctan(x[3]/x[2]))
            #amplitudes
            re = pqc_general(self.n_q)
            qc_par, p = re.build_qpc()
            p0 = 2*np.arctan(x[5]*c*d/x[2])
            p1 = np.arctan(x[3]/(x[0]*a)) + np.arctan(x[4]/(x[6]*b))
            p2 = np.arctan(x[3]/(x[0]*a)) - np.arctan(x[4]/(x[6]*b))
            p3 = 1/2*(np.arctan(x[1]/x[0]) + np.arctan(x[3]/x[2]) + np.arctan(x[5]/x[4]) + np.arctan(x[7]/x[6]))
            p4 = 1/2*(np.arctan(x[1]/x[0]) - np.arctan(x[3]/x[2]) + np.arctan(x[5]/x[4]) - np.arctan(x[7]/x[6]))
            p5 = 1/2*(np.arctan(x[1]/x[0]) - np.arctan(x[3]/x[2]) - np.arctan(x[5]/x[4]) + np.arctan(x[7]/x[6]))
            p6 = 1/2*(np.arctan(x[1]/x[0]) + np.arctan(x[3]/x[2]) - np.arctan(x[5]/x[4]) - np.arctan(x[7]/x[6]))
            qc =  qc_par.bind_parameters({p:[p0,p1,p2,p3,p4,p5,p6]})
        return qc

    def de_encoding(self, p):
        """
        Function that maps a vector of pqc parameters to real amplitudes of the quantum state prepared by the pqc.

        Input(s): 

        p: array of pqc parameters.

        Output(s):

        w: array of real amplitudes.
        """
        if self.n_q == 2:
            p = np.array(p)
            w = [np.cos(p[0]/2)*np.cos((p[1] + p[2])/2), np.cos(p[0]/2)*np.sin((p[1] + p[2])/2), np.sin(p[0]/2)*np.sin((p[1] - p[2])/2), np.sin(p[0]/2)*np.cos((p[1]-p[2])/2)]
        if self.n_q == 3:
            p = np.array(p)
            w0 = np.cos(p[0]/2)*np.cos((p[1] + p[2])/2)*np.cos((p[3]+p[4]+p[5]+p[6])/2)
            w1 = np.cos(p[0]/2)*np.cos((p[1] + p[2])/2)*np.sin((p[3]+p[4]+p[5]+p[6])/2)
            w2 = np.cos(p[0]/2)*np.sin((p[1] + p[2])/2)*np.cos((p[3]-p[4]-p[5]+p[6])/2)
            w3 = np.cos(p[0]/2)*np.sin((p[1] + p[2])/2)*np.sin((p[3]-p[4]-p[5]+p[6])/2)
            w4 = np.sin(p[0]/2)*np.sin((p[1] - p[2])/2)*np.cos((p[3]+p[4]-p[5]-p[6])/2)
            w5 = np.sin(p[0]/2)*np.sin((p[1] - p[2])/2)*np.sin((p[3]+p[4]-p[5]-p[6])/2)
            w6 = np.sin(p[0]/2)*np.cos((p[1] - p[2])/2)*np.cos((p[3]-p[4]+p[5]-p[6])/2)
            w7 = np.sin(p[0]/2)*np.cos((p[1] - p[2])/2)*np.sin((p[3]-p[4]+p[5]-p[6])/2)
            w = [w0, w1, w2, w3, w4, w5, w6, w7]
        return w

    def estereo(self, w):  #Inverse stereographic encoding, to make coordinates in R^n, coordinates of quantum state amplitude vector.
        """
        Inverse Stereographic projection
        Function that maps a vector of classical data to real amplitudes of a quantum state 

        Input(s): 

        w: array of classical data

        Output(s):

        x: array of real amplitudes.
        """
        m = np.size(w)
        x = np.zeros(m+1)
        norm_squared = 0
        for i in range(m):
            norm_squared = norm_squared + w[i]**2
        for i in range(m):
            x[i] = 2*w[i]/(norm_squared + 1)
        x[m] = (norm_squared - 1)/(norm_squared + 1)  
        return x

    def de_stereo(self, x): #Stereopgrahic projection of the input
        """
        Stereographic projection
        Function that maps a vector of real amplitudes of a quantum state to a vector in the real plane (classical data) 

        Input(s): 

        x: array of real amplitudes.

        Output(s):

        y: array of classical data
        """
        m = np.size(x)
        y = np.zeros(m-1)
        for i in range(m-1):
            y[i] = x[i]/(1-x[m-1])
        return y

    def new_encod(self,w):
        """
        
        Function that builds a pqc with real amplitudes, to encode classical data into a quantum circuit.

        Input(s): 

        w: array of classical data

        Output(s):

        real amplitude pqc encoding the input
        """
        x = self.estereo(w)
        return self.amp_enc(x)

    def new_de_encod(self,p):
        """
        Function that de-encodes the outputs from re-qgan, from real amplitude quantum states to classical data vectors.

        Input(s): 

        p: parameters of the pqc generated by re-qgan.

        Output(s):

        w: classical data generated by re-qgan
        """
        x = self.de_encoding(p)
        w = self.de_stereo(x)
        return w

    def disc_qc(self, x, qc_input, x_init = None):
        """
        Function that builds the pqc for the discriminator of re-qgan and returns the measurement from the ancilla qubit.

        Input(s): 

        x: discriminator's parameters (these are dynamically updated during training)

        qc_input: generator's pqc

        Output(s):

        mean_z: ancilla qubit measurement
        """
        x = np.array(x)
        qc =  self.qc_disc.bind_parameters({self.p_disc:x})
        qc1 = QuantumCircuit(self.n_q + 1)
        qr = qc1.qregs[0]
        if qc_input == None:
            qc1.initialize(x_init, [qr[i+1] for i in range(self.n_q)])
        else:
            qc1.append(qc_input.to_instruction(),[qr[i+1] for i in range(self.n_q)])
        qc1.append(qc.to_instruction(),[qr[i] for i in range(self.n_q + 1)])
        qc2 = qc1.decompose()
        cr = ClassicalRegister(self.n_q + 1, 'creg')
        qc2.add_register(cr)
        if self.mock_backend:
            qc2.measure(qr[0], cr[0])
            experiment = execute(qc2,backend=self.backend,shots=self.shots)
            results = experiment.result()
            counts = results.get_counts()
            N_0 = counts.get(np.binary_repr(0,self.n_q+1),0)
            #N_1 = counts.get(np.binary_repr(1,self.n_q+1),0)
            #mean_z = (N_1 - N_0)/self.shots
            mean_z = N_0/self.shots
        else:
            if self.noise_simulation == True:
                noise_model = self.noise_model(self.p1, self.p2, self.u, self.v) 
                qc2.measure(qr[0], cr[0])
                experiment = execute(qc2, backend=self.backend, shots=self.shots, noise_model=noise_model)
                results = experiment.result()
                counts = results.get_counts()
                try:
                    N_0 = counts.get(np.binary_repr(0,self.n_q+1),0)
                except:
                    N_0 = 0
                mean_z = N_0/self.shots
            if self.noise_simulation == False:
                op = 'I'*self.n_q + 'Z'
                mean_z = Statevector(qc2).expectation_value(Pauli(op))
                mean_z = (mean_z + 1)/2
        return mean_z
    #
    def obj_function_disc(self,x, store = True):
        """
        Discriminator's loss function

        Input(s): 

        x: discriminator's parameters (these are dynamically updated during training)

        Output(s):

        obj: value of the loss function
        """
        obj_D = 0.
        obj_G = 0
        if self.qc_real_build: #when we use the qpu
            for xr in self.x_real:
                qc_real = self.amp_enc(xr)
                obj_D += self.disc_qc(x, qc_real)
        else:
            for xr in self.x_real: #when we use the simulator
                # x are the parameters for the discriminator
                value = self.disc_qc(x, qc_input = None, x_init=xr)
                obj_D += (1-value)**2
                #storing the discriminator output with real data input 
                self.disc_output_real.append([value, self.epoch])
                # min of obj_D means disc_qc is one
        obj_D = 0.5*obj_D/len(self.x_real)
        #
        dim = len(self.x_real)
        for idx in range(dim):
            xg = self.x_gen[(2**self.n_q - 1)*idx:(idx+1)*(2**self.n_q - 1)]
            qc = self.qc_gen.bind_parameters({self.p_gen:xg})
            value = self.disc_qc(x, qc)
            obj_G += (value)**2
            #storing the discriminator output with gerenerated data input
            self.disc_output_gen.append([value, self.epoch])
        obj_G = 0.5*obj_G/dim
        #
        obj = obj_D + obj_G
        if store:
            self.res_real.append(obj)
        return obj

    def obj_function_gen(self, x, store = True):
        """
        Generator's loss function

        Input(s): 

        x: generator's parameters (these are dynamically updated during training)

        Output(s):

        obj_G: value of the loss function
        """
        dim = len(self.x_real)
        obj_G = 0
        for idx in range(dim):
            xg = x[(2**self.n_q - 1)*idx:(idx+1)*(2**self.n_q - 1)]
            qc = self.qc_gen.bind_parameters({self.p_gen:xg})
            obj_G += (1-self.disc_qc(self.x_dopt, qc))**2
        obj_G = obj_G/dim
        if store:
            self.res_gen.append(obj_G)
        return obj_G
    #
    def cnt(self, ii_ = [0],reset=False):
        ii_[0] += 1
        if reset:
            ii_[0] = 0
        return ii_[0]
    #
    def grad_obj_function_disc(self, x):
        """
        Discriminator's loss function for gradient based learning

        Input(s): 

        x: discriminator's parameters (these are dynamically updated during training)

        Output(s):

        grad: value of the loss function
        """
        grad = []
        x = np.array(x)
        val = self.obj_function_disc(x, store=False)
        for i in range(len(x)):
            #pi/2 phase
            x[i] += 0.5*np.pi
            val_p = self.obj_function_disc(x, store = False)
            #-pi/2 phase
            x[i] -= np.pi
            val_n = self.obj_function_disc(x, store = False)
            grad.append((val_p - val_n)/2.)
            #recover
            x[i] += 0.5*np.pi
        self.iter = self.cnt()
        self.res_real.append(val)
        print('Obj_D',self.iter, val)
        return np.array(grad)

    def grad_obj_function_gen(self, x):
        """
        Generator's loss function for gradient based learning

        Input(s): 

        x: generator's parameters (these are dynamically updated during training)

        Output(s):

        grad: value of the loss function
        """
        grad = []
        x = np.array(x)
        val = self.obj_function_gen(x, store=False)
        for i in range(len(x)):
            #pi/2 phase
            x[i] += 0.5*np.pi
            val_p = self.obj_function_gen(x, store = False)
            #-pi/2 phase
            x[i] -= 0.5*np.pi
            val_n = self.obj_function_gen(x, store = False)
            grad.append((val_p - val_n)/2.)
            #recover
            x[i] += 0.5*np.pi
        self.iter = self.cnt()
        self.res_real.append(val)
        print('Obj_G',self.iter, val)
        return np.array(grad)
    #
    def game(self,epochs):
        """
        qgan training

        Input(s): 

        epochs: Number of re-qgan training cycles

        Output(s):

        results: dictionary with re-qgan outputs, namely:
        results['disc_output_real'] = self.disc_output_real The discriminator's output when its input is real data
        results['disc_output_gen'] = self.disc_output_gen The discriminator's output when its input is data from the generator
        results['disc_training'] = self.inter_real The values of the discriminator's loss function over time
        results['gen_training'] = self.inter_gen The values of the generator's loss function over time
        results['gen_data'] = self.x_gen The generator's parameters at the last epoch
        results['gen_pca_img'] = w The generator's output at the last epoch
        results['w_gen_hist'] = w_hist The generator's outputs over time
        results['x_gen_hist'] = x_hist The generator's parameters over time
        results['gen_states'] = gen_states The generated quantum states over time
        results['quantum_dataset'] = self.x_real The parameters for the pqc that loads the real data 
        (note, here in the simulator, the quantum state is directly prepared by the simulator)
        """
        self.disc_output_real = []
        self.disc_output_gen = []
        np.random.seed()
        self.inter_real = []
        self.inter_gen = []
        x_hist = []
        dim_gen = len(self.x_real)*(2**self.n_q - 1)
        self.x_gen = np.random.uniform(-1.0*np.pi, +1.0*np.pi ,size=dim_gen)
        self.x_dopt = np.random.uniform(-1.0*np.pi, +1.0*np.pi ,size=(self.num_params))
        for i in range(epochs):
            #
            self.epoch = i
            print('#'*20)
            print('epoch :', i)
            self.res_real = []
            self.res_gen = []
            print('disc training')
            dim = self.num_params
            #
            x0 = self.x_dopt #paramaters for the discriminator
            #####################################
            # CMA-ES
            self.disc_input = "real"
            if self.opt == 'cma':
                options = {'maxfevals':self.opt_maxevals,'tolx': 1e-12, 'AdaptSigma': True, 'CMA_elitist':False, 'popsize': 4 + np.floor(2*np.log(dim))}
                res = cma.CMAEvolutionStrategy(x0, 0.1, options)
                while not res.stop():
                    solutions = res.ask()
                    costs = Parallel(n_jobs=7)(delayed(self.obj_function_disc)(x) for x in res.ask())
                    res.tell(solutions, costs)
                res.disp()
                self.x_dopt = res.result[0]
                self.inter_real.append(np.array(self.res_real))
            ####################################
            #Adam
            if self.opt == 'adam':
                self.iter = 0
                self.cnt(reset=True)
                output = climin.adam.Adam(wrt=x0, 
                                          fprime=self.grad_obj_function_disc,
                                          step_rate=0.1)
                for info in output:
                    if self.iter > self.opt_maxevals:
                        break
                self.inter_real.append(np.array(self.res_real))
            ###################################
            print('gen training')
            self.disc_input = "gen"
            ###################################
            dim  = 2**self.n_q - 1
            x0 = self.x_gen 
            ###################################
            # CMA-ES
            if self.opt == 'cma':
                options = {'maxfevals':self.opt_maxevals,'tolx': 1e-12, 'AdaptSigma': True, 'CMA_elitist':False} #, 'popsize': 4 + np.floor(2*np.log(dim))}
                res = cma.CMAEvolutionStrategy(x0, 0.1, options)
                while not res.stop():
                    solutions = res.ask()
                    costs = Parallel(n_jobs=5)(delayed(self.obj_function_gen)(x) for x in res.ask())
                    res.tell(solutions, costs)
                res.disp()
                #r = res.optimize(self.obj_function_gen).result
                self.x_gen = res.result[0]
                x_hist.append(res.result[0])
                self.inter_gen.append(np.array(self.res_gen))
            #####################################
            # ADAM
            if self.opt == 'adam':
                self.iter = 0
                self.cnt(reset=True)
                output = climin.adam.Adam(wrt=x0, 
                                          fprime=self.grad_obj_function_gen,
                                          step_rate=0.1)
                for info in output:
                    if self.iter > self.opt_maxevals:
                        break
                self.inter_real.append(np.array(self.res_gen))
            #####################################
        w = []
        dim = len(self.x_real)
        if self.mock_backend:
            for idx in range(dim):
                xg = self.x_gen[(2**self.n_q - 1)*idx:(idx+1)*(2**self.n_q - 1)]
                w.append(self.new_de_encod(xg))
        else:
            for idx in range(dim):
                xg = self.x_gen[(2**self.n_q - 1)*idx:(idx+1)*(2**self.n_q - 1)]
                qc = self.qc_gen.bind_parameters({self.p_gen:xg})
                ampls = np.real(Statevector(qc).data)
                if self.over_dim:
                    x = np.zeros(self.dim_input)
                    for i in range(self.dim_input):
                        x[i] = ampls[i]
                    w.append(self.de_stereo(x))
                else: 
                    w.append(self.de_stereo(ampls))
        # saving generated data in each epoch
        gen_states = []
        w_hist = []
        if self.mock_backend:
            for angle in self.x_hist:
                w_temp = []
                for idx in range(dim):
                    xg = angle[(2**self.n_q - 1)*idx:(idx+1)*(2**self.n_q - 1)]
                    w_temp.append(self.new_de_encod(xg))
        else:
            for angle in x_hist:
                w_temp = []
                for idx in range(dim):
                    xg = angle[(2**self.n_q - 1)*idx:(idx+1)*(2**self.n_q - 1)]
                    qc = self.qc_gen.bind_parameters({self.p_gen:xg})
                    ampls = np.real(Statevector(qc).data)
                    gen_states.append(ampls)
                    if self.over_dim:
                        x = np.zeros(self.dim_input)
                        for i in range(self.dim_input):
                            x[i] = ampls[i]
                        w_temp.append(self.de_stereo(x))
                    else: 
                        w_temp.append(self.de_stereo(ampls))
                w_hist.append(w_temp)
        #
        results = {}
        results['disc_output_real'] = self.disc_output_real
        results['disc_output_gen'] = self.disc_output_gen
        results['disc_training'] = self.inter_real
        results['gen_training'] = self.inter_gen
        results['gen_data'] = self.x_gen
        results['gen_pca_img'] = w
        results['w_gen_hist'] = w_hist
        results['x_gen_hist'] = x_hist
        results['gen_states'] = gen_states
        results['quantum_dataset'] = self.x_real
        if self.noise_simulation == True:
            results['noise_params'] = [self.p1, self.p2, self.u, self.v] 
        return results

        