import pennylane as qml
import numpy as np
from pennylane import I, PauliX, PauliY, PauliZ

I, X, Y, Z = I(0).matrix(), PauliX(0).matrix(), PauliY(0).matrix(), PauliZ(0).matrix()
one_pauli_matrices = [I, X, Y, Z]

two_pauli_matrices = []
for i in range(4):
    for j in range(4):
        two_pauli_matrices.append(np.kron(one_pauli_matrices[i], one_pauli_matrices[j]))


def get_random_probs(num_probs):
    """
    Generates random probabilities.
    :param num_probs: The number of probabilities to generate.
    :return: The generated probabilities.
    """
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


def get_qnode(n_wires):
    @qml.qnode(qml.device("default.mixed", n_wires))
    def circ(layers):
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
    PauliErrors(get_random_probs(16), wires=wires)
    qml.CNOT(wires)


def apply_spam_errors(num_wires):
    for wire in range(num_wires):
        SPAM(get_random_probs(4), wire)


def apply_X_gates(num_wires):
    for wire in range(num_wires):
        qml.X(wire)
