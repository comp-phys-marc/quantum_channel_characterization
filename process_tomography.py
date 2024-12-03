from qiskit_experiments.library.tomography import ProcessTomography
import matplotlib.pyplot as plt
import numpy as np
from circuit_qiskit import noise_model, circuit
from qiskit_aer import AerSimulator

if __name__ == '__main__':

    backend = AerSimulator(method='density_matrix', noise_model=noise_model)
    experiment_data = ProcessTomography(circuit).run(backend=backend).block_for_results()

    choi_matrix = experiment_data.analysis_results('state').value

    plt.matshow(np.real(choi_matrix))
    plt.show()
    plt.matshow(np.imag(choi_matrix))
    plt.show()
