from qiskit import QuantumRegister, ClassicalRegister

from protocol.connection_elements.ConnectionElement import ConnectionElement


class BaseEve(ConnectionElement):
    def __init__(self):
        self.eve_clone = None
        self.eve_measure = ClassicalRegister(1, "eve_measure")

    def init(self, n_bits: int, channel_size:int=1, seed=None):
        self.eve_clone = QuantumRegister(channel_size, "eve_clone")

    def cregs(self) -> list:
        return [self.eve_measure]

    def qregs(self) -> list:
        return [self.eve_clone]