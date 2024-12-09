import numpy as np
import scipy as sp
from unitary_circuit import get_circuit_matrix_repr


class NonUnitaryRepr:
    """
    The representation of the non-unitary circuit which may be in one embodiment a sum of Laplacians,
    or in another a sum of density matrices.
    """

    def __init__(self, unitary_systems=None, num_qubits=4):
        """
        Initializes a non-unitary circuit representation.
        :param unitary_systems: Subsystems which evolve unitarily.
        :param matrix_repr: The matrix representation used.
        :param num_qubits: The number of qubits in the system.
        """
        self.num_qubits = num_qubits
        if unitary_systems is not None:
            assert isinstance(unitary_systems, list)
            self.unitary_systems = unitary_systems
        else:
            self.unitary_systems = [np.array([[0 for j in range(0, i)] + [1 / (2 ** num_qubits)] +
                                        [0 for k in range(i + 1, 2 ** num_qubits)]
                                        for i in range(2 ** num_qubits)])]

    def sum(self):
        """
        Combines all the unitarily evolving subsystems into a non-unitary matrix repr by a sum.
        :return: None
        """
        sum = None
        for unitary_system in self.unitary_systems:
            if sum is None:
                sum = unitary_system
            else:
                sum += unitary_system

        self.unitary_systems = [sum]

    def apply_unitary_operator(self, op, repr):
        """
        Applies a unitary operator to the representation of the non-unitary circuit.
        :param op: The unitary operator to apply.
        :param repr: The matrix representation of the state.
        :return: The evolved matrix repr.
        """
        pass

    def apply_non_unitary_operator(self, op, repr):
        """
        Applies non-unitary Kraus operators to the representation of the non-unitary circuit.
        :param ops: The non-unitary operators to apply.
        :param repr: The matrix representation of the state.
        :return: The evolved matrix repr.
        """
        pass


class DensityMatrices(NonUnitaryRepr):

    def __init__(self, unitary_systems=None, num_qubits=4):
        """
        Accepts unitary systems which should be density matrices and initializes a non-unitary repr.
        :param unitary_systems: Systems which should be density matrices.
        :param num_qubits: The number of qubits in the system.
        """
        super().__init__(unitary_systems, num_qubits)

    def apply_unitary_method(self, op, sum=True):
        """
        Applies a unitary operator to the representation of the non-unitary circuit.
        :param op: The unitary operator to apply.
        param sum: Whether to sum the unitary systems after evaluation.
        :return: None
        """
        new_list = []
        for unitary_system in self.unitary_systems:
            new_unitary_system = op @ unitary_system @ np.transpose(op)
            new_list.append(new_unitary_system)
        self.unitary_systems = new_list
        if sum:
            self.sum()

    def apply_non_unitary_method(self, ops, sum=True):
        """
        Applies non-unitary Kraus operators to the representation of the non-unitary circuit.
        Weighted operators such as those in a Pauli error channel are assumed to have absorbed their weights.
        :param ops: The non-unitary operators to apply.
        :param repr: The matrix representation of the state.
        :param sum: Whether to sum the unitary systems after evaluation.
        :return: None
        """
        new_list = []
        for op in ops:
            for unitary_system in self.unitary_systems:
                new_unitary_system = op @ unitary_system @ np.transpose(op)
                new_list.append(new_unitary_system)
        self.unitary_systems = new_list
        if sum:
            self.sum()


def truncated_logm(truncate, diff):
    """
    Given the difference of an operator with the identity, calculates its truncated matrix logarithm.
    :param truncate: How far to expand the logarithm.
    :param diff: The difference of the operator with the identity.
    :return: The truncated matrix logarithm
    """
    logm = None
    for k in range(0, truncate):
        mul = None
        for i in range(k):
            if mul is None:
                mul = diff
            else:
                mul = np.matmul(diff, mul)
        if logm is None:
            logm = -mul / k
        else:
            logm -= mul / k
    return logm


class LaplacianMatrices(NonUnitaryRepr):

    def __init__(self, unitary_systems=None, num_qubits=4):
        """
        Accepts unitary systems which should be Laplacian matrices and initializes a non-unitary repr.
        :param unitary_systems: Systems which should be Laplacian matrices.
        :param num_qubits: The number of qubits in the system.
        """
        super().__init__(unitary_systems, num_qubits)

    def apply_unitary_method(self, op, sum=True, truncate=None):
        """
        Applies a unitary operator to the representation of the non-unitary circuit.
        :param op: The unitary operator to apply.
        :param sum: Whether to sum the unitary systems after evaluation.
        :param truncate: Whether to truncate the matrix logarithm if possible.
        :return: None
        """

        new_list = []
        for unitary_system in self.unitary_systems:
            diff = np.subtract(np.eye(2 ** self.num_qubits), op)
            if np.linalg.norm(diff, ord=2) < 1 and truncate is not None:
                new_unitary_system = truncated_logm(truncate, diff) \
                                      - unitary_system \
                                      + truncated_logm(truncate, np.subtract(np.eye(2 ** self.num_qubits),
                                                                             np.transpose(op)))
            else:
                new_unitary_system = sp.linalg.logm(op) - unitary_system + sp.linalg.logm(np.transpose(op))
            new_list.append(new_unitary_system)
        self.unitary_systems = new_list
        if sum:
            self.sum()

    def apply_non_unitary_method(self, ops, sum=True, truncate=None):
            """
            Applies non-unitary Kraus operators to the representation of the non-unitary circuit.
            Weighted operators such as those in a Pauli error channel are assumed to have absorbed their weights.
            :param ops: The non-unitary operators to apply.
            :param repr: The matrix representation of the state.
            :param sum: Whether to sum the unitary systems after evaluation.
            :param truncate: Whether to truncate the matrix logarithm if possible.
            :return: None
            """
            new_list = []
            for op in ops:
                for unitary_system in self.unitary_systems:
                    diff = np.subtract(np.eye(2 ** self.num_qubits), op)
                    if np.linalg.norm(diff, ord=2) < 1 and truncate is not None:
                        new_unitary_system = truncated_logm(truncate, diff) \
                                             - unitary_system \
                                             + truncated_logm(truncate, np.subtract(np.eye(2 ** self.num_qubits),
                                                                                    np.transpose(op)))
                    else:
                        new_unitary_system = sp.linalg.logm(op) - unitary_system + sp.linalg.logm(np.transpose(op))
                    new_list.append(new_unitary_system)
            self.unitary_systems = new_list
            if sum:
                self.sum()


def get_non_unitary_matrix_repr(num_wires, num_layers, cnot_error_probs, reset_error_probs, measurement_error_probs):
    """
    Calculates the non-unitary matrix representation of a circuit by density matrix simulation.
    :param circuit: The circuit.
    :return: The non-unitary matrix representation of the circuit.
    """
    get_circuit_matrix_repr(
        num_wires,
        num_layers,
        cnot_error_probs,
        reset_error_probs,
        measurement_error_probs,
        method=np.matmul,
        initial=None,
        error_method=draw_pauli_error,
        two_qubit_error_method=draw_pauli_error_string
    )