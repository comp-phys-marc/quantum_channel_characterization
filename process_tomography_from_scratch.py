# number of qubits
n = 10

# pauli basis
pauli_basis = ['I', 'X', 'Y', 'Z']

# number of Pauli strings for n qubits
num_strings = 4 ** n


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
    print(index_to_string(p, n))


