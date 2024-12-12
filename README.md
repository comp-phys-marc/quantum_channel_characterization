# Quantum Channel and State Graph Learning

In this repo we make use primarily of Qiskit and Jax to implement deep learning approaches to learning 
Pauli error rates on a circuit with many-body qubit interaction and dynamical decoupling.

The repo is broken into the following files, and generally into two components which can be decoupled. 

## model.py

This file contains the first component of the model which involves pooling, a fully connected NN with several 
hidden layers, and a softmax. This file also implements a loss function which is based on comparing process 
matrices reconstructed from the learned Pauli error rates to matrices obtained from process tomography, making
use of the second component of the model to calculate the process matrix.

## channels.py

This file contains functions that are useful for transforming channels between different representations and 
comparing them to each other.

## circuit.py

Circuit.py contains an implementation of the quantum circuit we chose to study using Xanadu's Pennylane quantum 
machine learning library.

## circuit_qiskit.py

While Pennylane has its advantages when we are interested in machine learning, Qiskit is arguably more feature-rich 
which became important for us as we needed to do process tomography. Qiskit Experiments has a process tomography 
implementation that we used to generate our training and benchmark data.

## data_utils.py 

This file contains utilities for saving and loading our data as well as transformting our data. We add support to json 
for complex values and we implement a translation from Qiskit's measurement counts to quantum expectation values.

## evalutation_utils.py

In here we implement the tools used to evaluate the performance of the various methods we compare to each other in this 
work.

## unitary_circuit.py

In unitary_circuit.py we implement a version of our circuit execution that is abstract in the sense that it can be
evaluated using various methods specified with a flag. This allows us to compare approaches involving the representation 
of quantum information with graph Laplacians, Kraus operators and density matrices.

## non_unitary_circuit.py

unitary_circuit.py can be used on its own in which case Pauli errors will be inserted by randomly drawing Paulis from 
an error probability distribution. However, in order to represent the non-unitary error channels in their entirety, it
is necessary to maintain and sum over a number of unitarily evolving subsystems. non_unitary_circuit.py implements the
logic for this use case.

## pauli_utils.py

This file contains the functions we use to perform operations related to Pauli algebra.

## process_tomography.py

process_tomgraphy.py contains the script and the function we use to generate our test and benchmark data, leveraging 
Qiskit.