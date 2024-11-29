# Qiskit imports
from fileinput import filename
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt

one_pauli_strings = ["I","X","Y","Z"]
two_pauli_strings = []
for pauli_1 in one_pauli_strings:
    for pauli_2 in one_pauli_strings:
        two_pauli_strings.append(pauli_1+pauli_2)




def get_circuit(num_wires, num_layers):
    qr = QuantumRegister(num_wires)
    cr = ClassicalRegister(num_wires)
    circuit = QuantumCircuit(qr, cr)
    for layer in range(num_layers):
        for wire in range(num_wires):
            circuit.x(qr[wire])
        apply_CNOT_layer(circuit, qr, num_wires, layer % 2)
    return circuit

def apply_CNOT_layer(circuit, qr, num_wires, layer_mod_2):
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
    noise_model = NoiseModel()
    error_reset = pauli_error(list(zip(one_pauli_strings, reset_error_probs)))
    error_cnot = pauli_error(list(zip(two_pauli_strings, cnot_error_probs)))
    error_measure = pauli_error(list(zip(one_pauli_strings, measurement_error_probs)))


    noise_model.add_all_qubit_quantum_error(error_cnot, ['cx'])
    noise_model.add_all_qubit_quantum_error(error_reset, "reset")
    noise_model.add_all_qubit_quantum_error(error_measure, "measure")

    return noise_model

def get_random_probs(num_probs):
    rands = np.random.rand(num_probs)
    return rands/np.linalg.norm(rands)


circuit = get_circuit(6,5)
circuit.draw(output="mpl", style='iqp', cregbundle=False)
plt.show()

cnot_probs = get_random_probs(16)
reset_probs = get_random_probs(4)
measurement_probs = get_random_probs(4)

noise_model = get_noise_model(cnot_probs, reset_probs, measurement_probs)
