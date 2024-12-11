# Qiskit imports
from fileinput import filename
from qiskit_aer.noise import NoiseModel, pauli_error

import random
from jax import numpy as jnp

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt


ONE_PAULI_STRINGS = ["I","X","Y","Z"]
TWO_PAULI_STRINGS = []
for pauli_1 in ONE_PAULI_STRINGS:
    for pauli_2 in ONE_PAULI_STRINGS:
        TWO_PAULI_STRINGS.append(pauli_1+pauli_2)


def get_circuit(num_wires, num_layers):
    """
    Generates the circuit for given number of wires and layers.
    :param num_wires: The number of wires / qubits.
    :param num_layers: The number of layers.
    :return: The circuit as a QuantumCircuit object.
    """
    qr = QuantumRegister(num_wires)
    cr = ClassicalRegister(num_wires)
    circuit = QuantumCircuit(qr, cr)
    for layer in range(num_layers):
        for wire in range(num_wires):
            circuit.x(qr[wire])
        apply_CNOT_layer(circuit, qr, num_wires, layer % 2)
    return circuit


def apply_CNOT_layer(circuit, qr, num_wires, layer_mod_2):
    """
    Applies the CNOT layer to the circuit.
    :param circuit: The circuit to add a CNOT layer to.
    :param qr: The quantum register.
    :param num_wires: The number of qubits.
    :param layer_mod_2: Whether this is an even or odd layer.
    :return: None
    """
    if num_wires % 2 == 0:
        final_wire = num_wires - layer_mod_2
        wrap_around = bool(layer_mod_2)
    else:
        final_wire = num_wires + layer_mod_2 - 1
        wrap_around = not bool(layer_mod_2)

    for wire in range(layer_mod_2, final_wire, 2):
        circuit.cx(qr[wire], qr[wire+1])

    if wrap_around:
        circuit.cx(qr[num_wires-1], qr[0])


def get_noise_model(cnot_error_probs, reset_error_probs, measurement_error_probs):
    """
    Constructs a noise model given gate error probabilities.
    :param cnot_error_probs: The probabilities of CNOT errors.
    :param reset_error_probs: The probabilities of reset errors.
    :param measurement_error_probs: The probabilities of measurement errors.
    :return: The noise model.
    """
    noise_model = NoiseModel()
    error_reset = pauli_error(list(zip(ONE_PAULI_STRINGS, reset_error_probs)))
    error_cnot = pauli_error(list(zip(TWO_PAULI_STRINGS, cnot_error_probs)))
    error_measure = pauli_error(list(zip(ONE_PAULI_STRINGS, measurement_error_probs)))

    noise_model.add_all_qubit_quantum_error(error_cnot, ['cx'])
    noise_model.add_all_qubit_quantum_error(error_reset, "reset")
    noise_model.add_all_qubit_quantum_error(error_measure, "measure")

    return noise_model


def get_random_probs(num_probs):
    """
    Generates random probabilities.
    :param num_probs: The number of probabilities to generate.
    :return: The generated probabilities.
    """
    rands = []
    sum = 0
    for i in range(num_probs):
        r = random.uniform(0, 1)
        rands.append(r)
        sum += r
    for j in range(len(rands)):
        rands[j] = rands[j] / sum
    return jnp.array(rands)


if __name__ == '__main__':

    circuit = get_circuit(3,3)
    circuit.draw(output="mpl", style='iqp', cregbundle=False)
    plt.show()
