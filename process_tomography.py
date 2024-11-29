from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_experiments.library.tomography import ProcessTomography
from qiskit.providers.basic_provider import BasicProvider
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr[0])
    circuit.measure(qr[0], cr[0])

    backend = BasicProvider().get_backend('basic_simulator')
    experiment_data = ProcessTomography(circuit).run(backend=backend).block_for_results()

    choi_matrix = experiment_data.analysis_results('state').value

    plt.matshow(np.real(choi_matrix))
    plt.show()
    plt.matshow(np.imag(choi_matrix))
    plt.show()