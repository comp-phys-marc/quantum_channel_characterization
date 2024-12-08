import json
import numpy as np
import math
from qiskit.quantum_info import Choi
from pauli_utils import index_to_string, THREE_EIGS, FOUR_EIGS, enumerate_observables


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Choi):
            return obj.data.tolist()
        elif isinstance(obj, complex):
            return f"{obj.real} + {obj.imag}i"
        return json.JSONEncoder.default(self, obj)


class ComplexDecoder(json.JSONDecoder):
    def __init__(self, **kwargs):
        kwargs.setdefault("object_hook", self.object_hook)
        super().__init__(**kwargs)

    def decode_complex(self, v):
        """Try to decode the complex number."""

        if isinstance(v, str) and 'i' in v and '+' in v:
            try:
                real_str = v.split('+')[0].strip()
                imag_str = v.split('+')[1][0:-1].strip()

                return complex(float(real_str), float(imag_str))
            except Exception as e:
                print(f"Could not parse {v} due to error {str(e)}")
        else:
            return v

    def object_hook(self, curr):
        """Try to decode the dict."""

        if isinstance(curr, dict):
            for k, v in curr.items():
                if isinstance(v, list) or isinstance(v, np.ndarray) or isinstance(v, dict):
                    curr[k] = self.object_hook(v)
                elif isinstance(v, str):
                    curr[k] = self.decode_complex(v)

        elif isinstance(curr, list) or isinstance(curr, np.ndarray):
            for i in range(len(curr)):
                if isinstance(curr[i], list) or isinstance(curr[i], np.ndarray) or isinstance(curr[i], dict):
                    curr[i] = self.object_hook(curr[i])
                if isinstance(curr[i], str):
                    curr[i] = self.decode_complex(curr[i])

        return curr


def bitstring_to_observable_eigenvalue(bitstring, pauli_string):
    """
    Converts a bitstring observed using a Pauli basis measurement to the corresponding eigenvalue.
    :param bitstring: The bitstring to convert.
    :param pauli_string: The Pauli string that specifies the measurement basis.
    :return: The observable eigenvalue.
    """
    # get Z-basis observable eigenvalue
    if len(pauli_string) <= 2:
        eigenvalue = -1
        if bitstring.count('1') % 2 == 0:
            eigenvalue = 1
        return eigenvalue

    # some case pre-computed for runtime optimization
    if len(pauli_string) == 3:
        eigenvectors = THREE_EIGS[1]
        eigenvalues = THREE_EIGS[0]
    elif len(pauli_string) == 4:
        eigenvectors = FOUR_EIGS[1]
        eigenvalues = FOUR_EIGS[0]
    else:
        [eigenvalues, eigenvectors] = enumerate_observables(len(pauli_string))  # this is slower

    outcome_vector = np.array([0. for _ in range(len(eigenvectors[0]))])
    outcome_vector[int(bitstring[0:len(pauli_string)], 2)] = 1.

    index = 0
    for eigenvector in eigenvectors:
        if (eigenvector == outcome_vector).all():
            break
        index += 1

    return eigenvalues[index]

    # TODO: do we want to convert to Pauli basis observable eigenvalues?


def expectation_matrix_from_counts(dataset="benchmarking", qubits=4, layers=4):
    """
    Builds expectation matrices from counts.
    :param dataset: The type of dataset to build expectation matrices from.
    :param qubits: The number of qubits.
    :param layers: The number of layers in the circuit.
    :return: The expectation matrices.
    """
    data = json.loads(json.loads(open(f"./data/{dataset}_dataset_{qubits}_qubits_{layers}_layers.json", "r").read()),
                      cls=ComplexDecoder)

    expectations_matrices = []

    for k, v in data.items():
        outcomes_arr = v['outcomes']
        i = 0
        expectations = np.array([[0.0 for _ in range(2 ** qubits)] for _ in range(4 ** qubits)])
        while i < (2 ** qubits) * (4 ** qubits):
            measurement = outcomes_arr[i]
            pauli_string = index_to_string(math.floor(i / (2 ** qubits)), qubits)  # TODO: is this how they are ordered? check assumption.
            counts = measurement['counts']
            total = 0
            total_weight = 0
            for bitstring, times_observed in counts.items():
                total += bitstring_to_observable_eigenvalue(bitstring, pauli_string) * times_observed
                total_weight += times_observed
            expectation = total / total_weight
            expectations[math.floor(i / (2 ** qubits))][i % (2 ** qubits)] = expectation  # TODO: check order assumption.
            i += 1
        expectations_matrices.append(expectations)

    return expectations_matrices


if __name__ == '__main__':
    expectations = expectation_matrix_from_counts()
    print(expectations)

    json_dict = dict()
    json_dict[0] = {
        "complex_matrix": [[complex(0.1, 0.2) for i in range(5)] for j in range(5)],
        "real_matrix": [[0.1 for i in range(5)] for j in range(5)]
    }

    dumped = json.dumps(json_dict, cls=NumpyEncoder)
    loaded = json.loads(dumped, cls=ComplexDecoder)

    assert np.array_equal(loaded['0']["complex_matrix"], json_dict[0]["complex_matrix"])
    assert np.array_equal(loaded['0']["real_matrix"], json_dict[0]["real_matrix"])
