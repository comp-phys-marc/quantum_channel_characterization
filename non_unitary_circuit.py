import jax.numpy as jnp
import scipy as sp

from circuit_qiskit import get_random_probs
from unitary_circuit import get_circuit_matrix_repr, correct_gate_dimensionality, laplacian_to_density_matrix, \
    density_matrix_to_laplacian
from pauli_utils import index_to_error_operator, index_to_string, LOOKUP
from qiskit.quantum_info import Kraus, Choi
from evaluation_utils import profile


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
            self.unitary_systems = [jnp.array([[0 for j in range(0, i)] + [1] +
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

    def apply_unitary_method(self, op):
        """
        Applies a unitary operator to the representation of the non-unitary circuit.
        :param op: The unitary operator to apply.
        :return: The evolved matrix repr.
        """
        pass

    def apply_non_unitary_method(self, op):
        """
        Applies non-unitary Kraus operators to the representation of the non-unitary circuit.
        :param ops: The non-unitary operators to apply.
        :return: The evolved matrix repr.
        """
        pass

    def single_error_application(self, error_probs, targets, num_qubits):
        """
        Applies a single-qubit error by simulating non-unitary dynamics.
        :param error_probs: The probability dist.
        :param targets: The target qubits.
        :param num_qubits: The number of qubits in the system.
        :param repr: The matrix repr of the circuit.
        :return: The evolved repr.
        """
        ops = []
        for i in range(len(error_probs)):
            pauli_op = index_to_error_operator(i, len(targets)) * error_probs[i]
            ops.append(correct_gate_dimensionality(pauli_op, targets, num_qubits))

        self.apply_non_unitary_method(ops)

        return self

    def two_error_applications(self, error_probs, targets, num_qubits):
        """
        Applies a two-qubit error by simulating non-unitary dynamics.
        :param error_probs: The probability dist from which to draw.
        :param targets: The target qubits.
        :param num_qubits: The number of qubits in the system.
        :param repr: The matrix repr of the circuit.
        :return: The evolved repr.
        """
        ops = []
        for i in range(len(error_probs)):
            pauli_ops_strs = index_to_string(i, len(targets))
            pauli_op = correct_gate_dimensionality(LOOKUP[pauli_ops_strs[0]], [targets[0]], num_qubits) @ \
                      correct_gate_dimensionality(LOOKUP[pauli_ops_strs[1]], [targets[1]], num_qubits) \
                      * error_probs[i]

            ops.append(pauli_op)

        self.apply_non_unitary_method(ops)

        return repr


class DensityMatrices(NonUnitaryRepr):

    def __init__(self, unitary_systems=None, num_qubits=4):
        """
        Accepts unitary systems which should be density matrices and initializes a non-unitary repr.
        :param unitary_systems: Systems which should be density matrices.
        :param num_qubits: The number of qubits in the system.
        """
        super().__init__(unitary_systems, num_qubits)
        self.seen_matrices = {}

    def apply_unitary_method(self, op, sum=False):
        """
        Applies a unitary operator to the representation of the non-unitary circuit.
        :param op: The unitary operator to apply.
        param sum: Whether to sum the unitary systems after evaluation.
        :return: None
        """
        new_list = []
        for unitary_system in self.unitary_systems:
            new_unitary_system = op @ unitary_system @ jnp.transpose(op)
            new_list.append(new_unitary_system)
            if hash_array(op) not in self.seen_matrices:
                self.seen_matrices[hash_array(op)] = {
                    'orig': op,
                    'applied_to': [unitary_system]
                }
            else:
                self.seen_matrices[hash_array(op)]['applied_to'].append(unitary_system)
        self.unitary_systems = new_list
        if sum:
            self.sum()

    def apply_non_unitary_method(self, ops, sum=False):
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
                new_unitary_system = op @ unitary_system @ jnp.transpose(op)
                new_list.append(new_unitary_system)
                if hash_array(op) not in self.seen_matrices:
                    self.seen_matrices[hash_array(op)] = {
                        'orig': op,
                        'applied_to': [unitary_system]
                    }
                else:
                    self.seen_matrices[hash_array(op)]['applied_to'].append(unitary_system)
        self.unitary_systems = new_list
        if sum:
            self.sum()


class Tape(NonUnitaryRepr):
    """
    Builds the Kraus representation of the circuit by recording the operations performed.
    """

    def __init__(self, unitary_systems=None, num_qubits=4):
        """
        Initializes a Tape object.
        :param unitary_systems: Subsystems which evolve unitarily are ignored by Tape.
        :param num_qubits: The number of qubits in the system.
        """
        super().__init__(unitary_systems, num_qubits)
        self.num_qubits = num_qubits
        self.unitary_systems = [jnp.array([[0 for j in range(0, i)] + [1] +
                                          [0 for k in range(i + 1, 2 ** num_qubits)]
                                          for i in range(2 ** num_qubits)])]

    def apply_unitary_method(self, op, sum=False):
        """
        Applies a unitary operator.
        :param op: The unitary operator to apply.
        param sum: Whether to sum the unitary systems after evaluation.
        :return: None
        """
        new_list = []
        for unitary_system in self.unitary_systems:
            new_unitary_system = op @ unitary_system
            new_list.append(new_unitary_system)
        self.unitary_systems = new_list
        if sum:
            self.sum()

    def apply_non_unitary_method(self, ops, sum=False):
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
                new_unitary_system = op @ unitary_system
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
        for i in range(k + 1):
            if mul is None:
                mul = diff
            else:
                mul = jnp.matmul(diff, mul)
        if logm is None:
            logm = -mul / (k + 1)
        else:
            logm -= mul / (k + 1)
    return logm


class LaplacianMatrices(NonUnitaryRepr):

    def __init__(self, unitary_systems=None, num_qubits=4):
        """
        Accepts unitary systems which should be Laplacian matrices and initializes a non-unitary repr.
        :param unitary_systems: Systems which should be Laplacian matrices.
        :param num_qubits: The number of qubits in the system.
        """
        super().__init__(unitary_systems, num_qubits)
        self.seen_matrices = {}

    def sum(self, truncate=None):
        """
        Sums the Laplacians by converting to density matrix form and converting back since
        there is no nice way to calculate the sum of exponentials with different exponents.
        :param truncate: How much to truncate the matrix exponentials.
        :return: None
        """
        sum = None
        for unitary_system in self.unitary_systems:
            density_matrix = laplacian_to_density_matrix(unitary_system, truncate=truncate)
            if sum is None:
                sum = density_matrix
            else:
                sum += density_matrix

        # convert back to Laplacian
        laplacian = density_matrix_to_laplacian(sum)
        self.unitary_systems = [laplacian]

    def single_error_application(self, error_probs, targets, num_qubits, truncate=None):
        """
        Applies a single-qubit error by simulating non-unitary dynamics.
        :param error_probs: The probability dist.
        :param targets: The target qubits.
        :param num_qubits: The number of qubits in the system.
        :param repr: The matrix repr of the circuit.
        :param truncate: How much to expand the power series.
        :return: The evolved repr.
        """
        ops = []
        for i in range(len(error_probs)):
            pauli_op = index_to_error_operator(i, len(targets)) * error_probs[i]
            ops.append(correct_gate_dimensionality(pauli_op, targets, num_qubits))

        self.apply_non_unitary_method(ops, truncate=truncate)

        return self

    def two_error_applications(self, error_probs, targets, num_qubits, truncate=None):
        """
        Applies a two-qubit error by simulating non-unitary dynamics.
        :param error_probs: The probability dist from which to draw.
        :param targets: The target qubits.
        :param num_qubits: The number of qubits in the system.
        :param repr: The matrix repr of the circuit.
        :param truncate: How much to expand the power series.
        :return: The evolved repr.
        """
        ops = []
        for i in range(len(error_probs)):
            pauli_ops_strs = index_to_string(i, len(targets))
            pauli_op = correct_gate_dimensionality(LOOKUP[pauli_ops_strs[0]], [targets[0]], num_qubits) @ \
                      correct_gate_dimensionality(LOOKUP[pauli_ops_strs[1]], [targets[1]], num_qubits) \
                      * error_probs[i]

            ops.append(pauli_op)

        self.apply_non_unitary_method(ops, truncate=truncate)

        return repr

    def apply_unitary_method(self, op, sum=False, truncate=None):
        """
        Applies a unitary operator to the representation of the non-unitary circuit.
        :param op: The unitary operator to apply.
        :param sum: Whether to sum the unitary systems after evaluation.
        :param truncate: Whether to truncate the matrix logarithm if possible.
        :return: None
        """

        new_list = []
        for unitary_system in self.unitary_systems:
            hash = hash_array(op)
            if len(self.seen_matrices) > 0 and hash in self.seen_matrices:
                new_unitary_system = (self.seen_matrices[hash]['logm'] -
                                      unitary_system + self.seen_matrices[hash]['logm_inv'])
                self.seen_matrices[hash_array(op)]['added_to'].append(unitary_system)
            else:
                diff = jnp.subtract(jnp.eye(2 ** self.num_qubits), op)
                if jnp.linalg.norm(diff, ord=2) < 1 and truncate is not None:
                    truncated = truncated_logm(truncate, diff)
                    truncated_inv = truncated_logm(truncate, jnp.subtract(jnp.eye(2 ** self.num_qubits),
                                                                                 jnp.transpose(op)))
                    new_unitary_system = truncated - unitary_system + truncated_inv
                    self.seen_matrices[hash_array(op)] = {
                        'logm': truncated,
                        'logm_inv': truncated_inv,
                        'orig': op,
                        'added_to': [unitary_system]
                    }
                else:
                    lgm = sp.linalg.logm(op)
                    lgm_inv = sp.linalg.logm(jnp.transpose(op))
                    new_unitary_system = lgm - unitary_system + lgm_inv
                    self.seen_matrices[hash_array(op)] = {
                        'logm': lgm,
                        'logm_inv': lgm_inv,
                        'orig': op,
                        'added_to': [unitary_system]
                    }
            new_list.append(new_unitary_system)
        self.unitary_systems = new_list
        if sum:
            self.sum()

    def apply_non_unitary_method(self, ops, sum=False, truncate=None):
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
                    hash = hash_array(op)
                    if len(self.seen_matrices) > 0 and hash in self.seen_matrices:
                        new_unitary_system = (self.seen_matrices[hash]['logm'] -
                                              unitary_system + self.seen_matrices[hash]['logm_inv'])
                        self.seen_matrices[hash_array(op)]['added_to'].append(unitary_system)
                    else:
                        diff = jnp.subtract(jnp.eye(2 ** self.num_qubits), op)
                        if jnp.linalg.norm(diff, ord=2) < 1 and truncate is not None:
                            truncated = truncated_logm(truncate, diff)
                            truncated_inv = truncated_logm(truncate, jnp.subtract(jnp.eye(2 ** self.num_qubits),
                                                                                 jnp.transpose(op)))
                            new_unitary_system = truncated - unitary_system + truncated_inv
                            self.seen_matrices[hash_array(op)] = {
                                'logm': truncated,
                                'logm_inv': truncated_inv,
                                'orig': op,
                                'added_to': [unitary_system]
                            }
                        else:
                            lgm = sp.linalg.logm(op)
                            lgm_inv = sp.linalg.logm(jnp.transpose(op))
                            new_unitary_system = lgm - unitary_system + lgm_inv
                            self.seen_matrices[hash_array(op)] = {
                                'logm': lgm,
                                'logm_inv': lgm_inv,
                                'orig': op,
                                'added_to': [unitary_system]
                            }
                    new_list.append(new_unitary_system)
            self.unitary_systems = new_list
            if sum:
                self.sum()


def hash_array(arr, precision=3):
    hash = ''
    for elem in arr:
        if not isinstance(elem, jnp.ndarray) and not isinstance(elem, list):
            hash += str(elem).split('.')[0] + '.'
            if len(str(elem).split('.')) > 1:
                dec = str(elem).split('.')[1]
                i = 0
                while i < len(dec) and i < precision:
                    hash += dec[i]
                    i += 1
        else:
            hash += hash_array(elem)
    return hash


@profile
def get_non_unitary_matrix_repr(
        num_wires,
        num_layers,
        cnot_error_probs,
        reset_error_probs,
        measurement_error_probs,
        type="density_matrix"
    ):
    """
    Calculates the non-unitary matrix representation of a circuit by density matrix simulation.
    :param num_wires: The number of wires in the circuit.
    :param num_layers: The number of layers in the circuit.
    :param cnot_error_probs: The probabilities of CNOT errors.
    :param reset_error_probs: The probabilities of reset errors.
    :param measurement_error_probs: The probabilities of measurement errors.
    :param type: The type of representation.
    :return: The non-unitary matrix representation of the circuit.
    """

    def unitary_evolution_method(op, repr):
        if type == "laplacian":
            repr.apply_unitary_method(op, truncate=1)
        else:
            repr.apply_unitary_method(op)
        return repr

    def single_error_method(error_probs, targets, num_wires, repr):
        if type == "laplacian":
            repr.single_error_application(error_probs, targets, num_wires, truncate=1)
        else:
            repr.single_error_application(error_probs, targets, num_wires)
        return repr

    def two_qubit_error_method(error_probs, targets, num_wires, repr):
        if type == "laplacian":
            repr.two_error_applications(error_probs, targets, num_wires, truncate=1)
        else:
            repr.two_error_applications(error_probs, targets, num_wires)
        return repr

    if type == "density_matrix":
        initial = DensityMatrices(num_qubits=num_wires)
    elif type == "laplacian":
        initial = LaplacianMatrices(num_qubits=num_wires)
    else:
        initial = Tape(num_qubits=num_wires)

    repr = get_circuit_matrix_repr(
        num_wires,
        num_layers,
        cnot_error_probs,
        reset_error_probs,
        measurement_error_probs,
        method=unitary_evolution_method,
        initial=initial,
        error_method=single_error_method,
        two_qubit_error_method=two_qubit_error_method
    )

    return repr


if __name__ == "__main__":
    # get error probs
    cnot_probs = get_random_probs(16)
    reset_probs = get_random_probs(4)
    measurement_probs = get_random_probs(4)

    print("Building Choi matrix")

    # get Choi matrix
    repr = get_non_unitary_matrix_repr(
        2,
        2,
        cnot_probs,
        reset_probs,
        measurement_probs,
        type="tape"
    )

    kraus = Kraus(repr.unitary_systems)
    choi = Choi(kraus)

    print("Evolving density matrix")

    # simulate evolution using density matrix approach
    density_matrix = get_non_unitary_matrix_repr(
        2,
        2,
        cnot_probs,
        reset_probs,
        measurement_probs,
        type="density_matrix"
    )

    print("Evolving graph Laplacian")

    # simulate evolution using graph Laplacian approach
    repr = get_non_unitary_matrix_repr(
        2,
        2,
        cnot_probs,
        reset_probs,
        measurement_probs,
        type="laplacian"
    )

    # direct test of advantage

    # comparison of entire circuit execution ab-initio
    @profile
    def re_execute_density_matrix_circuit(density_matrix):
        for k, v in density_matrix.seen_matrices.items():
            for target in v['applied_to']:
                res = jnp.matmul(jnp.matmul(v['orig'], target), v['orig'])

    @profile
    def re_execute_laplacian_circuit(laplacian):
        for k, v in laplacian.seen_matrices.items():
            for target in v['added_to']:
                res = v['logm'] + -target + v['logm_inv']


    print("Re-playing density matrix evolution")
    re_execute_density_matrix_circuit(density_matrix)
    print("Re-playing Laplacian evolution")
    re_execute_laplacian_circuit(repr)

    # get data on each gate
    @profile
    def apply_orig(orig, target):
        orig @ target @ orig

    @profile
    def apply_logm(logm, target):
        logm - target + logm

    target = jnp.eye(2 ** repr.num_qubits)
    for k, v in repr.seen_matrices.items():
        orig = v['orig']
        apply_orig(orig, target)

    target = jnp.eye(2 ** repr.num_qubits)
    for k, v in repr.seen_matrices.items():
        logm = v['logm']
        apply_logm(logm, target)
