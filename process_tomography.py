from qiskit_experiments.library.tomography import ProcessTomography
from circuit_qiskit import get_random_probs, get_noise_model, get_circuit
from qiskit_aer import AerSimulator
import json
from data_utils import NumpyEncoder


def generate_dataset(n_qubits, depth, n_samples):
    json_dict = {}
    circuit = get_circuit(n_qubits, depth)

    for i in range(n_samples):
        print(f"Run {i}")

        cnot_probs = get_random_probs(16)
        reset_probs = get_random_probs(4)
        measurement_probs = get_random_probs(4)

        noise_model = get_noise_model(cnot_probs, reset_probs, measurement_probs)

        backend = AerSimulator(method='density_matrix', noise_model=noise_model)
        experiment_data = ProcessTomography(circuit).run(backend=backend).block_for_results()

        choi_matrix = experiment_data.analysis_results('state').value

        counts_list = list(map(lambda c: {'counts': c['counts']}, experiment_data.data()))

        json_dict[str(i)] = {
            "cnot_probs": cnot_probs,
            "reset_probs": reset_probs,
            "measurement_probs": measurement_probs,
            "choi_matrix": choi_matrix,
            "outcomes": counts_list
        }

    return json_dict


if __name__ == '__main__':
    print("Additional training data generation")
    for m in range(4, 5):
        print(f"Qubits: {m}")
        for n in range(2, 5):
            print(f"Layers: {n}")
            f = open(f"data/additional_training_dataset_{m}_qubits_{n}_layers.json", "w")

            json_dict = generate_dataset(m, n, 100)

            json.dump(json.dumps(json_dict, cls=NumpyEncoder), f)
            f.close()
    #
    # print("Benchmark data generation")
    # for m in range(2, 5):
    #     print(f"Qubits: {m}")
    #     for n in range(2, 5):
    #         print(f"Layers: {n}")
    #         f = open(f"data/benchmarking_dataset_{m}_qubits_{n}_layers.json", "w")
    #
    #         json_dict = generate_dataset(m, n, 100)
    #
    #         json.dump(json.dumps(json_dict, cls=NumpyEncoder), f)
    #         f.close()
    #
    # print("Training data generation")
    # for m in range(2, 5):
    #     print(f"Qubits: {m}")
    #     for n in range(2, 5):
    #         print(f"Layers: {n}")
    #         f = open(f"data/training_dataset_{m}_qubits_{n}_layers.json", "w")
    #
    #         json_dict = generate_dataset(m, n, 100)
    #
    #         json.dump(json.dumps(json_dict, cls=NumpyEncoder), f)
    #         f.close()
