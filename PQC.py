""" this class uses the new ansatz that generates an n-qubit arbitrary quantum state with real amplitudes.  
 |\psi > = c_0 |0..0> + ... + c_{2^n}|11...1>, where c_i \in R. 

 The parameterized quantum circuit for a 3-qubit system follows the structure:

 q_0--Ry--x-------------x-------------
          |             |            
 q_1--Ry--x--Ry--x-------------x------
                 |      |      |      
 q_2--Ry---------x--Ry--x--Ry--x--Ry--

 The number of parameters is equal to the number of independent amplitudes, i.e. 2^n -1. The number of entangling gates, in this case CX-gates, is 2^{n+1} - 4 (I have to check this).  This version of the quantum circuit is not optimal in the number of CX-gates.  
"""
import numpy as np
from qiskit import *
from scipy import *
import itertools
from qiskit.circuit import QuantumCircuit, ParameterVector, Gate


class pqc_general:
    def __init__(self, qubits):
        self.n_q = len(qubits)  # number of qubits
        self.qubits = qubits
    #

    def build_qpc(self, mirror=False, st_to_st=False):
        """
        Function that builds the re-pqc for n qubits, with 2^(n) - 1 parameters.

        Input(s):

        n_q :  number of qubits

        Output(s):

        qc: re-pqc
        p: paramater vector for the re-pqc

        """
        n_q = self.n_q
        num_params = 2**n_q - 1
        if st_to_st:
            num_params = num_params + n_q - 1
        instructions = self.pqc_instructions()
        #
        p = ParameterVector('p', num_params)
        qc = QuantumCircuit(n_q)
        idx = 0
        if mirror:
            instructions = instructions[::-1]
        for inst in instructions:
            if inst[0] == 'ry':
                qc.ry(p[idx], inst[1])
                idx += 1
            if inst[0] == 'cx':
                qc.cx(inst[1], inst[2])
        if st_to_st:
            for qb in self.qubits[:self.n_q-1]:
                qc.ry(p[idx], qb)
                idx += 1
        return qc, p

    def pqc_instructions(self,):
        """
        function outputs "how" each block is going to be built, based on the number of qubits, 
        to avoid repeating RY gates  

        """
        insts = []
        #

        def block(insts, x):
            # one qubit block
            if len(x) == 1:
                insts.append(['ry', x[0], None])
                return insts
            # two-qubit block
            if len(x) == 2:
                insts.append(['ry', x[1], None])
                insts.append(['cx', x[0], x[1]])
                insts.append(['ry', x[1], None])
                return insts
            if len(x) > 2:
                new_len = len(x) - 1
                insts = block(insts, x[-1*new_len:])
                insts.append(['cx', x[0], x[-1]])
                insts = block(insts, x[-1*new_len:])
                return insts
        #
        couple = []
        for q_idx in self.qubits:
            couple.append(q_idx)
            insts = block(insts, couple)
        return insts
