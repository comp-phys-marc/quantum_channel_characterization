import functools

import pennylane as qml
from jax import numpy as jnp
import jax

I = jnp.eye(2)

X = jnp.array([[0.j,1+0.j],
               [1+0.j,0.j]])

Y = jnp.array([[0.j,-1.j],
               [1.j,0.j]])

Z = jnp.array([[1.+0.j,0.j],
               [0.j,-1+0.j]])
one_pauli_matrices = [I, X, Y, Z]

two_pauli_matrices = []
for i in range(4):
    for j in range(4):
        two_pauli_matrices.append(jnp.kron(one_pauli_matrices[i], one_pauli_matrices[j]))



def get_qnode(n_wires, layers, reset_probs, cnot_probs, measurement_probs):
    @qml.qnode(qml.device("default.mixed", n_wires))
    def circ(input_matrix):
        qml.QubitDensityMatrix(input_matrix, wires=range(n_wires))

        apply_spam_errors(n_wires, reset_probs)

        for layer in range(layers):
            apply_X_gates(n_wires)
            apply_CNOT_layer(n_wires, layer%2, cnot_probs)

        apply_spam_errors(n_wires, measurement_probs)

        return qml.state()
    return circ


def apply_CNOT_layer(num_wires, layer_mod_2, probs):
    if num_wires % 2 == 0:
        final_wire = num_wires - layer_mod_2
        wrap_around = bool(layer_mod_2)
    else:
        final_wire = num_wires + layer_mod_2 - 1
        wrap_around = not bool(layer_mod_2)

    for wire in range(layer_mod_2, final_wire, 2):
        wires=(wire, wire + 1)
        apply_noisy_CNOT(wires, probs)

    if wrap_around:
        apply_noisy_CNOT((num_wires-1, 0), probs)


def apply_noisy_CNOT(wires, probs):
    kraus = [jnp.sqrt(p) * pauli for p, pauli in zip(probs, two_pauli_matrices)]
    qml.QubitChannel(kraus, wires=wires)
    qml.CNOT(wires)


def apply_spam_errors(num_wires, probs):
    kraus = [jnp.sqrt(p) * pauli for p, pauli in zip(probs, one_pauli_matrices)]
    for wire in range(num_wires):
        qml.QubitChannel(kraus , wire)


def apply_X_gates(num_wires):
    for wire in range(num_wires):
        qml.X(wire)

@functools.partial(jax.jit, static_argnums=(3, 4))
def choi_calc_method(reset_probs, cnot_probs, measurement_probs, num_qubits, num_layers):
    qnode = jax.jit(get_qnode(num_qubits, num_layers, reset_probs, cnot_probs, measurement_probs))
    dm_rows = 2**num_qubits
    #TODO change to scan loops
    @jax.jit
    def get_choi_value_for_indices(row, col):
        input_dm = jnp.zeros((dm_rows, dm_rows)).at[row, col].set(1)
        return jnp.kron(input_dm, qnode(input_dm))

    choi_matrix = jnp.zeros((dm_rows**2, dm_rows**2))

    for i in range(dm_rows):
        for j in range(dm_rows):
            choi_matrix += get_choi_value_for_indices(i, j)

    return choi_matrix
