## Notes

### Nov 27th 

To get qiskit 0.6.0 (with tomography) working on M3 Mac:

- Had to demote to networkx 2.3 for compatibility
- Had to use x86 conda channel to get python3.7 in order to support fractions.gcd used in networkx==0.23
- Python 3.7 not compatible with pennylane (any version as far as I can tell)
- Therefore we'll have to execute our scripts separately, pass QASM back and forth

### Nov 28th 

- Updated to qiskit 1.3.0 and installed qiskit-experiments