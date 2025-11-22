from qiskit import QuantumRegister, ClassicalRegister

from protocol.connection_elements.ConnectionElement import ConnectionElement


class BaseEve(ConnectionElement):
    def __init__(self):
        self.eve_clone = QuantumRegister(1, "eve_clone")
        self.eve_measure = ClassicalRegister(1, "eve_measure")
        self._memory = None

    def cregs(self) -> list:
        return [self.eve_measure]

    def qregs(self) -> list:
        return [self.eve_clone]