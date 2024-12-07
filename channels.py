import json
from qiskit_aer import Aer
from qiskit.quantum_info import SparsePauliOp, Kraus
from evaluation_utils import profile
import matplotlib.pyplot as plt
import numpy as np
from circuit_qiskit import get_circuit
from data_utils import ComplexDecoder


def compare_choi_matrices(choi_one, choi_two):
    """
    Compares two Choi matrices using 8 matrix norms.
    :param choi_one: The first Choi matrix.
    :param choi_two: The second Choi matrix.
    :return: The norms of the difference of the two Choi matrices.
    """
    assert(choi_one.shape == choi_two.shape)
    diff = np.subtract(choi_one, choi_two)
    return [
        np.linalg.norm(diff, ord='fro'),   # Frobenius norm
        np.linalg.norm(diff, ord='nuc'),   # Nuclear norm
        np.linalg.norm(diff, ord=np.inf),  # max(sum(abs(x), axis=1))
        np.linalg.norm(diff, ord=-np.inf), # min(sum(abs(x), axis=1))
        np.linalg.norm(diff, ord=1),       # max(sum(abs(x), axis=0))
        np.linalg.norm(diff, ord=-1),      # min(sum(abs(x), axis=0))
        np.linalg.norm(diff, ord=2),       # 2-norm (largest singular value)
        np.linalg.norm(diff, ord=-2),      # smallest singular value
    ]


def plot_choi(choi_matrix):
    """
    Plots the Choi matrix, its real and complex components separately.
    :param choi_matrix: The matrix to plot.
    :return: None
    """
    plt.matshow(np.real(choi_matrix))
    plt.show()
    plt.matshow(np.imag(choi_matrix))
    plt.show()


@profile
def get_full_pauli_basis(n):
    """
    Returns the full pauli basis for n qubits.
    :param n: Number of qubits.
    :return: The full Pauli basis.
    """
    basis_elements = []

    def index_to_string(i, n):
        digits = []
        while i:
            digits.append(str(int(i % 4)))
            i //= 4
        unpadded_string = ("".join(digits[::-1])
                           .replace('0', 'I')
                           .replace('1', 'X')
                           .replace('2', 'Y')
                           .replace('3', 'Z'))

        return "".join(['I' for k in range(n - len(unpadded_string))]) + unpadded_string

    # all the measurement bases
    for p in range(4 ** n):
        basis_elements.append(index_to_string(p, n))

    return basis_elements


@profile
def kraus_operator_in_pauli_basis(kraus_operator):
    """
    Decomposes an operator into the Pauli basis.
    :param kraus_operator: a single Kraus operator, not the whole channel.
    :return: The Pauli operators and their coefficients.
    """
    return SparsePauliOp.from_operator(kraus_operator)


@profile
def super_operator_from_pauli_operator(pauli_operator):
    """
    Constructs a superoperator from a weighted sum of Paulis.
    :param pauli_operator: The SparsePauliOp to convert into a superoperator.
    :return: The constructed superoperator.
    """
    num_qubits = len(pauli_operator.paulis[0])
    coeffs = pauli_operator.coeffs
    basis = get_full_pauli_basis(num_qubits)  # TODO: implement inverse lookup so don't have to gen full basis
    super_operator = [[0 for r in range(len(basis))] for s in range(len(basis))]
    for i in range(len(coeffs)):
        m = basis.index(str(pauli_operator.paulis[i]))
        for j in range(len(coeffs)):
            n = basis.index(str(pauli_operator.paulis[j]))
            super_operator[m][n] = coeffs[i] * np.conjugate(coeffs[j])
    return super_operator


@profile
def kraus_channel_as_super_operator(kraus_channel):
    """
    Transforms a Kraus channel into a superoperator.
    :param kraus_channel: The Kraus channel to convert.
    :return: The superoperator.
    """
    super_operator = None
    for kraus_op in kraus_channel.data:
        pauli_op = kraus_operator_in_pauli_basis(kraus_op)
        if super_operator is None:
            super_operator = np.array(super_operator_from_pauli_operator(pauli_op))
        else:
            super_operator = np.add(super_operator, super_operator_from_pauli_operator(pauli_op))
    return super_operator


@profile
def circuit_to_unitary(circuit):
    """
    Returns the unitary of a circuit by simulating it.
    :param circuit: The circuit to convert.
    :return: The unitary representation.
    """
    backend = Aer.get_backend('unitary_simulator')
    result = backend.run(circuit).result()
    unitary = result.get_unitary(circuit)
    return unitary


@profile
def unitary_to_kraus_operator(unitary):
    """
    Transforms a unitary into a Kraus representation.
    :param circuit: The circuit to transform.
    :return: The superoperator.
    """
    return Kraus([unitary])


@profile
def circuit_to_super_operator(circuit):
    """
    Transforms a circuit into a superoperator by multiple steps.
    :param circuit: The circuit to transform.
    :return: The super operator.
    """
    unitary = circuit_to_unitary(circuit)
    kraus = unitary_to_kraus_operator(unitary)
    super_operator = kraus_channel_as_super_operator(kraus)
    return super_operator


if __name__ == "__main__":
    channel_ops = Kraus([np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, -1]])])
    super_operator = kraus_channel_as_super_operator(channel_ops)
    print(super_operator)

    for q in range(2, 5):
        print(f"qubits: {q}")
        for d in range(2, 5):
            print(f"depth: {d}")
            circuit = get_circuit(q, d)
            super_operator = circuit_to_super_operator(circuit)
            print(super_operator)

    data = json.loads(json.loads(open("./data/training_dataset_2_qubits_2_layers.json", "r").read()), cls=ComplexDecoder)

    norms = compare_choi_matrices(np.array(data["0"]["choi_matrix"]), np.array(data["1"]["choi_matrix"]))
    print(norms)
