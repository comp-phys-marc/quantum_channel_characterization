from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, execute
import qiskit.tools.qcvv.tomography as tomo
from qiskit.tools.visualization import plot_state

if __name__ == '__main__':

    qr = QuantumRegister(4)
    cr = ClassicalRegister(4)
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[0])
    circuit.measure(qr[0], cr[0])

    tomography_set = tomo.process_tomography_set([0])
    tomography_circuits = tomo.create_tomography_circuits(circuit, qr, cr, tomography_set)

    backend = Aer.get_backend('qasm_simulator')
    tomography_job = execute(tomography_circuits, backend=backend, shots=100)
    tomography_result = tomography_job.result()
    process_data = tomo.tomography_data(tomography_result, circuit.name, tomography_set)

    choi_fit = tomo.fit_tomography_data(process_data, options={'trace': 4})
    plot_state(choi_fit)
