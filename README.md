## Notes

To get qiskit 0.6.0 (with tomography) working on M3 Mac:

- Had to demote to networkx 2.3 for compatibility
- Had to use x86 conda channel to get python3.7 in order to support fractions.gcd used in networkx==0.23
- Python 3.7 not compatible with pennylane (any version as far as I can tell)
- Therefore we'll have to execute our scripts separately, pass QASM back and forth