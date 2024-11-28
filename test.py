from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, execute

qr = QuantumRegister(4)
cr = ClassicalRegister(4)
circuit = QuantumCircuit(qr, cr)
circuit.x(qr[0])
circuit.measure(qr[0], cr[0])

backend = Aer.get_backend('qasm_simulator')
tomography_job = execute(circuit, backend=backend, shots=100)
tomography_result = tomography_job.result()

