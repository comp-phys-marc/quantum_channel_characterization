from jax import numpy as jnp
I, X, Y, Z = jnp.eye(2), jnp.array([[0,1],[1,0]]), jnp.array([[0,-1j],[1j,0]]), jnp.array([[1,0],[0,-1]])
one_pauli_matrices = [I, X, Y, Z]

def get_pauli_Matrices(n=2):
    all_paulis = [I,X,Y,Z]
    for _ in jnp.arange(n-1):
        next_paulis = []
        for current_pauli in all_paulis:
            next_paulis += [jnp.kron(pauli, current_pauli) for pauli in one_pauli_matrices]
        all_paulis = next_paulis
    return jnp.array([pauli.reshape((n**4,)) for pauli in all_paulis])

def get_pauli_strings(n=2):
    all_paulis = ["I","X","Y","Z"]
    for _ in range(n-1):
        next_pauli_str = []
        for current_pauli in all_paulis:
            next_pauli_str += [current_pauli + pauli for pauli in ["I","X","Y","Z"]]
        all_paulis = next_pauli_str
    return all_paulis