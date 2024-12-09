import json
import random
import scipy as sp
import networkx as nx
import numpy as np
from functools import reduce
from channels import unitary_to_kraus_operator, kraus_channel_as_super_operator
from data_utils import ComplexDecoder
from pauli_utils import index_to_string, PAULI_X, index_to_error_operator, LOOKUP
from evaluation_utils import profile
from circuit_qiskit import get_random_probs
from data_utils import NumpyEncoder
from qiskit.quantum_info.operators.channel import Choi, SuperOp


CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])

# some frequently used matrices pre-computed for runtime optimization

CNOT_10 = np.array([[1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0]])
CNOT_20 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0]])
CNOT_30 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])


def correct_gate_dimensionality(gate, targets, num_qubits):
    """
    Correct a gate's dimensionality so it may be applied to a circuit.
    Note that two qubit gates must be applied to adjacent qubits.
    :param targets: The target qubits.
    :param gate: The gate whose dimensionality to correct.
    :return: The corrected gate.
    """

    if len(targets) == 1:
        return reduce(
            lambda state, d_qubit: np.kron(state, d_qubit),
            [(np.eye(2) if q not in targets else gate) for q in range(num_qubits)]
        )
    else:
        return get_CNOT_matrix(targets[0], targets[1], num_qubits)


def draw_pauli_error(pauli_probs, target_size):
    """
    Given probabilities of Pauli errors, draws a random Pauli.
    :param pauli_probs: The probabilities of Pauli errors.
    :param target_size: The number of target qubits.
    :return: The Pauli matrix drawn.
    """
    rand = random.uniform(0, 1)

    for i in range(len(pauli_probs)):
        sum = 0
        for j in range(i + 1):
            sum += pauli_probs[j]
        if rand < sum:
            return index_to_error_operator(i, target_size)


def draw_pauli_error_string(pauli_probs, target_size):
    """
    Given probabilities of Pauli errors, draws a random Pauli string.
    :param pauli_probs: The probabilities of Pauli errors.
    :param target_size: The number of target qubits.
    :return: The Pauli string.
    """
    rand = random.uniform(0, 1)

    for i in range(len(pauli_probs)):
        sum = 0
        for j in range(i + 1):
            sum += pauli_probs[j]
        if rand < sum:
            return index_to_string(i, target_size)


def apply_two_qubit_errors(errors):
    """
    Applies two-qubit errors as two single qubit operators.
    :param error_one: The first error, with its dim corrected.
    :param error_two: The second error, with its dim corrected.
    :return: The matrix repr of the circuit.
    """
    repr = np.matmul(errors[0], repr)
    repr = np.matmul(errors[1], repr)
    return repr


@profile
def get_circuit_matrix_repr(
        num_wires,
        num_layers,
        cnot_error_probs,
        reset_error_probs,
        measurement_error_probs,
        method=np.matmul,
        initial=None,
        error_method=draw_pauli_error,
        apply_error_method=np.matmul,
        two_qubit_error_method=draw_pauli_error_string,
        apply_two_qubit_error_method=apply_two_qubit_errors
    ):
    """
    Constructs the matrix representation of a circuit with errors included as randomly drawn Pauli.
    Can construct the Laplacian or the unitary representation depending on the method provided.
    :param num_wires: The number of wires in the circuit.
    :param num_layers: The number of layers in the circuit.
    :param cnot_error_probs: The probabilities of CNOT errors.
    :param reset_error_probs: The probabilities of reset errors.
    :param measurement_error_probs: The probabilities of measurement errors.
    :param initial: The initial matrix repr.
    :param method: The method to use to apply the gates.
    :param apply_error_method: The method to use in order to apply errors.
    :param error_method: The method to use in order to choose singel qubit errors.
    :param two_qubit_error_method: The method to use in order to choose two qubit errors.
    :return: The matrix representation of the circuit.
    """
    if initial is None:
        repr = np.eye(2 ** num_wires)
    else:
        repr = initial
    for qubit in range(num_wires):
        reset_error = correct_gate_dimensionality(error_method(reset_error_probs, 1),
                                                  [qubit], num_wires)
        repr = apply_error_method(reset_error, repr)

    for layer in range(num_layers):
        for wire in range(num_wires):
            flip = correct_gate_dimensionality(PAULI_X, [wire], num_wires)
            repr = method(flip, repr)

        repr = apply_CNOT_layer(
            repr,
            num_wires,
            layer % 2,
            cnot_error_probs,
            method,
            two_qubit_error_method,
            apply_two_qubit_error_method
        )

    for qubit in range(num_wires):
        meas_error = correct_gate_dimensionality(error_method(measurement_error_probs, 1),
                                                  [qubit], num_wires)
        repr = apply_error_method(meas_error, repr)

    return repr


def apply_CNOT_layer(repr, num_wires, layer_mod_2, cnot_error_probs, method, error_method, apply_error_method):
    """
    Applies the CNOT layer to the matrix representation of the circuit.
    :param circuit: The matrix representation to transform by a CNOT layer.
    :param num_wires: The number of qubits.
    :param layer_mod_2: Whether this is an even or odd layer.
    :param cnot_error_probs: The probabilities of CNOT errors.
    :param method: The method to use to apply the CNOTs and errors.
    :param error_method: The method to use in order to choose errors.
    :param apply_error_method: The method to use in order to apply errors.
    :return: The matrix representation of the circuit with the CNOT layer applied.
    """
    if num_wires % 2 == 0:
        final_wire = num_wires - layer_mod_2
        wrap_around = bool(layer_mod_2)
    else:
        final_wire = num_wires + layer_mod_2 - 1
        wrap_around = not bool(layer_mod_2)

    for wire in range(layer_mod_2, final_wire, 2):
        repr = method(correct_gate_dimensionality(CNOT, [wire, wire + 1], num_wires), repr)

        pauli_error_str = error_method(cnot_error_probs, 2)

        repr = apply_error_method([
            correct_gate_dimensionality(LOOKUP[pauli_error_str[0]], [wire], num_wires),
            correct_gate_dimensionality(LOOKUP[pauli_error_str[1]], [wire + 1], num_wires)
        ], repr)

    if wrap_around:
        if num_wires == 2:
            repr = method(CNOT_10, repr)
        elif num_wires == 3:
            repr = method(CNOT_20, repr)
        elif num_wires == 4:
            repr = method(CNOT_30, repr)
        else:
            raise NotImplementedError("Only up to 4 qubits supported.")

        pauli_error_str = error_method(cnot_error_probs, 2)

        repr = apply_error_method([
            correct_gate_dimensionality(LOOKUP[pauli_error_str[0]], [num_wires - 1], num_wires),
            correct_gate_dimensionality(LOOKUP[pauli_error_str[1]], [0], num_wires)
        ], repr)

    return repr


def get_CNOT_matrix(source, target, num_qubits):
    """
    Returns a CNOT matrix for the given source, target and number of qubits.
    :param source: The source qubit.
    :param target: The target qubit.
    :param num_qubits: The number of qubits.
    :return: The CNOT gate.
    """

    cx_matrix = [[0. for _ in range(2 ** num_qubits)] for _ in range(2 ** num_qubits)]

    for i, row in enumerate(cx_matrix):
        label = f'{i:0{num_qubits}b}'
        if label[source] == '1':
            label = label[0:target] + '0' if label[target] == '1' else '1' + label[target + 1:]
        one_position = int(label, 2)
        row[one_position] = 1.

    return np.array(cx_matrix)


def adjacency_matrix_to_laplacian(graph):
    """
    Converts a networkx graph to a Laplacian matrix.
    :param graph: The graph to convert.
    :return: The Laplacian matrix.
    """
    laplacian = sp.sparse.csr_matrix.toarray(nx.laplacian_matrix(nx.from_numpy_array(graph)))
    return laplacian


def apply_unitary(unitary, laplacian):
    """
    Performs the equivalent of a unitary evolution on the Laplacian representation. Assumes beta=1.
    :param laplacian: The Laplacian representing the state that evolves with unitary dynamics.
    :param unitary: The unitary to apply.
    :return: The Laplacian after unitary evolution.
    """
    return sp.linalg.logm(unitary) - laplacian + sp.linalg.logm(np.transpose(unitary))


def laplacian_to_density_matrix(laplacian, beta=1, truncate=None):
    """
    Converts a Laplacian matrix to a density matrix.
    :param laplacian: The Laplacian to convert.
    :param beta: The inverse temperature.
    :param truncate: How much to truncate the matrix exponential.
    :return: The density matrix.
    """
    if truncate is None:
        return (sp.linalg.expm(-beta * laplacian)) / np.trace(sp.linalg.expm(-beta * laplacian))
    else:
        expm = None
        for p in range(truncate):
            mul = None
            for q in range(p):
                if mul is None:
                    mul = (-beta * laplacian)
                else:
                    mul = np.matmul(-beta * laplacian, mul)
            if expm is None:
                expm = mul / sp.math.factorial(p)
            else:
                expm += mul / sp.math.factorial(p)
        return expm / np.trace(expm)


def pauli_probs_to_laplacian(cnot_probs, reset_probs, measurement_probs, num_qubits, num_layers):
    """
    Creates a Laplacian representing a circuit with the provided noise model.
    :param pauli_probs: The probabilities of Pauli errors.
    :return: The Laplacian.
    """
    laplacian = np.array([[0 for j in range(0, i)] + [1 / (2 ** n)] +
                          [0 for k in range(i, 2 ** n)] for i in range(2 ** n)])

    return get_circuit_matrix_repr(
        num_qubits,
        num_layers,
        cnot_probs,
        reset_probs,
        measurement_probs,
        method=apply_unitary,
        initial=laplacian
    )


def pauli_probs_to_super_operator(cnot_probs, reset_probs, measurement_probs, num_qubits, num_layers):
    """
    Calculates a superoperator from a set of Pauli error probabilities.
    :param cnot_probs: The probability of a CNOT error.
    :param reset_probs: The probability of a reset error.
    :param measurement_probs: The probability of a measurement error.
    :param num_qubits: The number of qubits in the circuit.
    :param num_layers: The number of layers in the circuit.
    :return:
    """
    unitary = get_circuit_matrix_repr(
        num_qubits,
        num_layers,
        cnot_probs,
        reset_probs,
        measurement_probs
    )

    super_operator = kraus_channel_as_super_operator(unitary_to_kraus_operator(unitary))
    return super_operator


def generate_data(num_qubits, layers, n_samples):
    """
    Generates data for the given circuit parameters.
    :param num_qubits: The number of qubits in the circuit.
    :param layers: The number of layers in the circuit.
    :param n_samples: The number of samples.
    :return: None
    """
    data = {}
    for i in range(n_samples):
        cnot_probs = get_random_probs(16)
        reset_probs = get_random_probs(4)
        measurement_probs = get_random_probs(4)

        super_op = pauli_probs_to_super_operator(cnot_probs, reset_probs, measurement_probs, num_qubits, layers)
        choi = Choi(SuperOp(super_op))

        data[str(i)] = {
            'cnot_probs': cnot_probs,
            'reset_probs': reset_probs,
            'measurement_probs': measurement_probs,
            'choi_matrix': choi
        }

    f = open(f"data/direct_training_dataset_4_qubits_{layers}_layers.json", "w")

    json.dump(json.dumps(data, cls=NumpyEncoder), f)
    f.close()


if __name__ == "__main__":
    # smoketest for quick Choi matrix generation method
    data_from_tomograhpy = json.loads(json.loads(open("./data/training_dataset_4_qubits_2_layers.json", "r").read()),
                      cls=ComplexDecoder)
    data_from_direct_method = json.loads(json.loads(open("./data/direct_training_dataset_4_qubits_2_layers.json", "r")
                                                    .read()), cls=ComplexDecoder)

    assert (np.array(data_from_tomograhpy['0']['choi_matrix']).shape ==
            np.array(data_from_direct_method['0']['choi_matrix']).shape)


    # generate 4-qubit data quickly
    for layers in range(2, 5):
        generate_data(4, layers, 100)

    # number of qubits
    n = 2

    # initial matrix representation
    laplacian = np.array([[0 for j in range(0, i)] + [1 / (2 ** n)] +
                          [0 for k in range(i + 1, 2 ** n)] for i in range(2 ** n)])

    # load some example error probabilities
    data = json.loads(json.loads(open("./data/training_dataset_2_qubits_2_layers.json", "r").read()),
                      cls=ComplexDecoder)

    # simulate one shot with a graph Laplacian
    laplacian = get_circuit_matrix_repr(
        n,
        2,
        data['0']['cnot_probs'],
        data['0']['reset_probs'],
        data['0']['measurement_probs'],
        method=apply_unitary,
        initial=laplacian
    )
