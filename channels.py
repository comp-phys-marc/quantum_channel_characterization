import json
from qiskit_aer import Aer
from qiskit.quantum_info import SparsePauliOp, Kraus
from evaluation_utils import profile
import matplotlib.pyplot as plt
import jax.numpy as jnp
from circuit_qiskit import get_circuit
from data_utils import ComplexDecoder
from pauli_utils import index_to_string


def compare_choi_matrices(choi_one, choi_two):
    """
    Compares two Choi matrices using 8 matrix norms.
    :param choi_one: The first Choi matrix.
    :param choi_two: The second Choi matrix.
    :return: The norms of the difference of the two Choi matrices.
    """
    assert(choi_one.shape == choi_two.shape)
    diff = jnp.subtract(choi_one, choi_two)
    return [
        jnp.linalg.norm(diff, ord='fro'),   # Frobenius norm
        jnp.linalg.norm(diff, ord='nuc'),   # Nuclear norm
        jnp.linalg.norm(diff, ord=jnp.inf),  # max(sum(abs(x), axis=1))
        jnp.linalg.norm(diff, ord=-jnp.inf), # min(sum(abs(x), axis=1))
        jnp.linalg.norm(diff, ord=1),       # max(sum(abs(x), axis=0))
        jnp.linalg.norm(diff, ord=-1),      # min(sum(abs(x), axis=0))
        jnp.linalg.norm(diff, ord=2),       # 2-norm (largest singular value)
        jnp.linalg.norm(diff, ord=-2),      # smallest singular value
    ]


def plot_choi(choi_matrix):
    """
    Plots the Choi matrix, its real and complex components separately.
    :param choi_matrix: The matrix to plot.
    :return: None
    """
    plt.matshow(jnp.real(choi_matrix))
    plt.show()
    plt.matshow(jnp.imag(choi_matrix))
    plt.show()


@profile
def get_full_pauli_basis(n):
    """
    Returns the full pauli basis for n qubits.
    :param n: Number of qubits.
    :return: The full Pauli basis.
    """
    basis_elements = []

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
            super_operator[m][n] = coeffs[i] * jnp.conjugate(coeffs[j])
    return jnp.array(super_operator)


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
            super_operator = jnp.array(super_operator_from_pauli_operator(pauli_op))
        else:
            super_operator = jnp.add(super_operator, super_operator_from_pauli_operator(pauli_op))
    return super_operator


@profile
def super_operator_to_choi(super_operator):
    """
    Converts a superoperator into a Choi matrix.
    :param super_operator: The super operator.
    :return: The Choi matrix.
    """
    d = int(jnp.round(jnp.sqrt(super_operator.shape[0])))
    assert super_operator.shape == (d * d, d * d)

    s = jnp.reshape(super_operator, (d, d, d, d))
    c = jnp.swapaxes(s, 1, 2)
    return jnp.reshape(c, (d * d, d * d))


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
    channel_ops = Kraus([jnp.array([[1, 0], [0, 1]]), jnp.array([[0, 1], [1, 0]]), jnp.array([[1, 0], [0, -1]])])
    super_operator = kraus_channel_as_super_operator(channel_ops)
    print(super_operator)
    choi = super_operator_to_choi(super_operator)
    print(choi)

    for q in range(2, 5):
        print(f"qubits: {q}")
        for d in range(2, 5):
            print(f"depth: {d}")
            circuit = get_circuit(q, d)
            super_operator = circuit_to_super_operator(circuit)
            print(super_operator)

    data = json.loads(json.loads(open("./data/training_dataset_2_qubits_2_layers.json", "r").read()),
                      cls=ComplexDecoder)

    norms = compare_choi_matrices(jnp.array(data["0"]["choi_matrix"]), jnp.array(data["1"]["choi_matrix"]))
    print(norms)
