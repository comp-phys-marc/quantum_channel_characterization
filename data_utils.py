import json
import numpy as np
from qiskit.quantum_info import Choi

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


def expectation_matrix_from_counts(dataset="benchmarking", qubits=2, layers=2):
    """
    Builds expectation matrices from counts.
    :param dataset: The type of dataset to build expectation matrices from.
    :param qubits: The number of qubits.
    :param layers: The number of layers in the circuit.
    :return: The expectation matrices.
    """
    data = json.loads(json.loads(open(f"./data/{dataset}_dataset_{qubits}_qubits_{layers}_layers.json", "r").read()),
                      cls=ComplexDecoder)



if __name__ == '__main__':
    json_dict = dict()
    json_dict[0] = {
        "complex_matrix": [[complex(0.1, 0.2) for i in range(5)] for j in range(5)],
        "real_matrix": [[0.1 for i in range(5)] for j in range(5)]
    }

    dumped = json.dumps(json_dict, cls=NumpyEncoder)
    loaded = json.loads(dumped, cls=ComplexDecoder)

    assert np.array_equal(loaded['0']["complex_matrix"], json_dict[0]["complex_matrix"])
    assert np.array_equal(loaded['0']["real_matrix"], json_dict[0]["real_matrix"])
