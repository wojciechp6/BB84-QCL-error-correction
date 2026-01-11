import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter

from protocol.connection_elements.ConnectionElement import ConnectionElement
from protocol.input_parameter import BoolInputParameter


class Bob(ConnectionElement):
    def __init__(self):
        self.base_p =  BoolInputParameter("bob_base")
        self.measure = ClassicalRegister(1, "bob_measure")

    def init(self, n_bits: int, channel_size=1, seed=None):
        random = np.random.RandomState(seed)
        self.base_p.values = random.randint(2, size=n_bits)

    def qc(self, channel: QuantumRegister, i, ctx: dict) -> QuantumCircuit:
        qc = QuantumCircuit(channel, self.measure, name="Bob")
        qc.ry(self.base_p * -np.pi/2, channel[0])
        qc.measure(channel[0], self.measure)
        if i is not None:
            qc.assign_parameters({"bob_base": self.base_p.values[i]}, inplace=True)
        return qc

    def cregs(self) -> list:
        return [self.measure]

    def input_params(self) -> list:
        return [self.base_p]
