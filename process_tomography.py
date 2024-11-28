from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, execute
import qiskit.tools.qcvv.tomography as tomo
from qiskit.qasm import Qasm
from qiskit.tools.visualization import plot_state

program = Qasm(filename="./intermediate.qasm").parse()

qr = QuantumRegister(4)
cr = ClassicalRegister(4)
circuit = QuantumCircuit(qr, cr)

for node in program.children:
    if node.__class__.__name__ == 'CustomUnitary':
        if node.name != 'cx':
            getattr(circuit, node.name)(qr[node.bitlist.children[0].children[1].value])
        else:
            getattr(circuit, node.name)(
                qr[node.bitlist.children[0].children[1].value],
                qr[node.bitlist.children[1].children[1].value]
            )

tomography_set = tomo.process_tomography_set([0, 1, 2, 3])
tomography_circuits = tomo.create_tomography_circuits(circuit, qr, cr, tomography_set)

backend = Aer.get_backend('qasm_simulator')
tomography_job = execute(tomography_circuits, backend=backend, shots=100)
tomography_result = tomography_job.result()
process_data = tomo.tomography_data(tomography_result, circuit.name, tomography_set)

choi_fit = tomo.fit_tomography_data(process_data, options={'trace': 4})
plot_state(choi_fit)
