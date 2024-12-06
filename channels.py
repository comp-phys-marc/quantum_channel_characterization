import numpy as np
from qiskit_aer import Aer
from qiskit.quantum_info import SparsePauliOp, Kraus
from circuit_qiskit import get_circuit
from evaluation_utils import profile

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

    circuit = get_circuit(2, 2)
    super_operator = circuit_to_super_operator(circuit)
    print(super_operator)
