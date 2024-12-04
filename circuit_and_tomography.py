import pennylane as qml
import numpy as np
from pennylane import I, PauliX, PauliY, PauliZ
# import jax

I, X, Y, Z = I(0).matrix(), PauliX(0).matrix(), PauliY(0).matrix(), PauliZ(0).matrix()
one_pauli_matrices = [I, X, Y, Z]
one_pauli_strings = ["I","X","Y","Z"]

two_pauli_strings = []
two_pauli_matrices = []
for i in range(4):
    for j in range(4):
        two_pauli_strings.append(one_pauli_strings[i]+one_pauli_strings[j])
        two_pauli_matrices.append(np.kron(one_pauli_matrices[i], one_pauli_matrices[j]))

def get_random_probs(num_probs):
    rands = np.random.rand(num_probs)
    return rands/np.sum(rands)


class SPAM(qml.QubitChannel):
    def __init__(self, probs, wire):
        kraus_matrices = [np.sqrt(p) * pauli for p, pauli in zip(probs, one_pauli_matrices)]
        super(SPAM, self).__init__(kraus_matrices, wires=wire)
        self.name = "QubitChannel"

class PauliErrors(qml.QubitChannel):
    def __init__(self, probs, wires):
        kraus_matrices = [np.sqrt(p) * pauli for p, pauli in zip(probs, two_pauli_matrices)]
        super(PauliErrors, self).__init__(kraus_matrices, wires=wires)
        self.name = "QubitChannel"


def get_qnode(n_wires, layers):
    @qml.qnode(qml.device("default.mixed", n_wires))
    def circ():
        #TODO change initial state
        apply_spam_errors(n_wires)

        for layer in range(layers):
            apply_X_gates(n_wires)
            apply_CNOT_layer(n_wires, layer%2)

        apply_spam_errors(n_wires)

        return qml.state()
    return circ

def apply_CNOT_layer(num_wires, layer_mod_2):
    if num_wires % 2 == 0:
        final_wire = num_wires - layer_mod_2
        wrap_around = bool(layer_mod_2)
    else:
        final_wire = num_wires + layer_mod_2 - 1
        wrap_around = not bool(layer_mod_2)

    for wire in range(layer_mod_2, final_wire, 2):
        wires=(wire, wire + 1)
        apply_noisy_CNOT(wires)

    if wrap_around:
        apply_noisy_CNOT((num_wires-1, 0))


def apply_noisy_CNOT(wires):
    PauliErrors(get_random_probs(16), wires=wires)  # TODO change
    qml.CNOT(wires)


def apply_spam_errors(num_wires):
    for wire in range(num_wires):
        SPAM(get_random_probs(4), wire)

def apply_X_gates(num_wires):
    for wire in range(num_wires):
        qml.X(wire)

# my_circ = get_qnode(5, 5)
# my_circ()two_pauli_matrices
# my_circ.qtape
for matrix in two_pauli_matrices:
    if matrix.dtype is complex:
        print("complex")
    else:
        print("real")
    print(matrix)
    print()
#==================================================================================================
# TOMOGRAPHY

# def apply_tomogrpahy(state_from_circuit, num_wires):
#     @qml.qnode("default.mixed", num_wires)
#     def measurement_circ(measurement):
#         qml.QubitDensityMatrix(state_from_circuit)
#         qml.expval(measurement)
#
#     # jax.lax.scan() TODO this may be worthwhile later
#     num_paulis = 4**num_wires
#     choi_matrix = np.zeros((num_paulis,num_paulis))
#     Hamiltonians = []
#     while len(Hamiltonians < num_paulis):
#         pass
#
#     for i in range(num_paulis):
#         for j in range(num_paulis):



-1/2[[ 0  1  0  0]
 [ 1  0  0  0]
 [ 0  0  0 -1]
 [ 0  0 -1  0]]

+1/2[[0. 1. 0. 0.]
 [1. 0. 0. 0.]
 [0. 0. 0. 1.]
 [0. 0. 1. 0.]]


-1/2 [[ 1.  0.  0.  0.]
 [ 0.  1.  0.  0.]
 [ 0.  0. -1. -0.]
 [ 0.  0. -0. -1.]]

+1/2[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]


[[ 0.+0.j  0.-1.j  0.+0.j  0.-0.j]
 [ 0.+1.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.-0.j -0.+0.j  0.+1.j]
 [ 0.+0.j  0.+0.j -0.-1.j -0.+0.j]]




