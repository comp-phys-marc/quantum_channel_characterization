import numpy as np

PAULI_X = np.array([[0, 1],
                    [1, 0]])
PAULI_Y = np.array([[0, -1j],
                    [1j, 0]])
PAULI_Z = np.array([[1, 0],
                    [0, -1]])
LOOKUP = {
    'I': np.eye(2),
    'X': PAULI_X,
    'Y': PAULI_Y,
    'Z': PAULI_Z,
}


def enumerate_observables(num_qubits):
    """
    Generates the eigenvector, eigenvalue pairs for the given number of qubits.
    :param num_qubits: The number of qubits.
    :return: The pairs of eigenvectors and eigenvalues.
    """
    observable = None
    for i in range(num_qubits):
        if observable is None:
            observable = PAULI_Z
        else:
            observable = np.kron(PAULI_Z, observable)

    eigenvalues, eigenvectors = np.linalg.eigh(observable)

    return [eigenvalues, eigenvectors]


THREE_EIGS = enumerate_observables(3)
FOUR_EIGS = enumerate_observables(4)


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


def index_to_error_operator(i, target_size):
    digits = []
    while i:
        digits.append(str(int(i % 4)))
        i //= 4
    unpadded_len = 0
    error = None
    for digit in reversed(digits):
        unpadded_len += 1
        if int(digit) == 0:
            op = np.eye(2 ** target_size)
        elif int(digit) == 1:
            op = PAULI_X
        elif int(digit) == 2:
            op = PAULI_Y
        elif int(digit) == 3:
            op = PAULI_Z
        if error is None:
            error = op
        else:
            error = np.kron(op, error)

    for j in range(target_size - unpadded_len):
        if error is None:
            error = np.eye(2 ** target_size)
        else:
            error = np.kron(np.eye(2 ** target_size), error)

    return error


if __name__ == "__main__":
    pass