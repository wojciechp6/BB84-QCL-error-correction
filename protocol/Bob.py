import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter

from protocol.connection_elements.ConnectionElement import ConnectionElement


class Bob(ConnectionElement):
    def __init__(self):
        self.base_p = Parameter("bob_base")
        self.bases = None
        self.measure = ClassicalRegister(1, "bob_measure")

    def init(self, n_bits: int, seed=None):
        random = np.random.RandomState(seed)
        self.bases = random.randint(2, size=n_bits)

    def qc(self, channel: QuantumRegister, i, ctx: dict) -> QuantumCircuit:
        qc = QuantumCircuit(channel, self.measure, name="Bob")
        qc.ry(self.base_p * -np.pi/2, 0)
        qc.measure(channel, self.measure)
        if i is not None:
            qc.assign_parameters({"bob_base": self.bases[i]}, inplace=True)
        return qc

    def cregs(self) -> list:
        return [self.measure]

    def input_params(self) -> list:
        return [self.base_p]

    def input_values(self) -> list:
        return [self.bases]