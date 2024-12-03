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
