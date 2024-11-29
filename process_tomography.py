from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_experiments.library.tomography import ProcessTomography
from qiskit.providers.basic_provider import BasicProvider
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[0])
    circuit.x(qr[1])
    circuit.x(qr[2])
    # circuit.x(qr[3])
    # circuit.x(qr[4])
    circuit.cx(qr[0], qr[1])
    circuit.cx(qr[1], qr[2])
    # circuit.cx(qr[2], qr[3])
    # circuit.cx(qr[3], qr[4])
    # circuit.cx(qr[4], qr[0])
    circuit.measure(qr[0], cr[0])
    circuit.measure(qr[1], cr[1])
    circuit.measure(qr[2], cr[2])
    # circuit.measure(qr[3], cr[3])
    # circuit.measure(qr[4], cr[4])

    backend = BasicProvider().get_backend('basic_simulator')
    experiment_data = ProcessTomography(circuit).run(backend=backend).block_for_results()

    choi_matrix = experiment_data.analysis_results('state').value

    plt.matshow(np.real(choi_matrix))
    plt.show()
    plt.matshow(np.imag(choi_matrix))
    plt.show()
