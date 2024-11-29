import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

def get_qnode(n_wires):
    @qml.qnode(qml.device("default.mixed", n_wires))
    def circ(layers):
        apply_spam_errors(n_wires)

        for layer in range(layers):
            apply_X_gates(n_wires)
            apply_CNOT_layer(n_wires, layer%2)

        apply_spam_errors(n_wires)

        return qml.state()
    return circ

def apply_CNOT_layer(num_wires, layer_mod_2):
    if num_wires % 2 == 0:
        final_wire = num_wires - layer_mod_2
        wrap_around = bool(layer_mod_2)
    else:
        final_wire = num_wires + layer_mod_2
        wrap_around = not bool(layer_mod_2)

    for wire in range(layer_mod_2, final_wire, 2):
        wires=(wire, wire + 1)
        apply_noisy_CNOT(wires)

    if wrap_around:
        apply_noisy_CNOT((num_wires-1, 0))


def apply_noisy_CNOT(wires):
    qml.QubitChannel([np.eye(4)], wires=wires, id=r"\Lambda")
    qml.CNOT(wires)


def apply_spam_errors(num_wires):
    for wire in range(num_wires):
        qml.QubitChannel([np.eye(2)], wire, id="SPAM") # TODO change

def apply_X_gates(num_wires):
    for wire in range(num_wires):
        qml.X(wire)

fig, ax = qml.draw_mpl(circ)(10)
fig.savefig("circ.jpeg")