import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit_aer import Aer

from protocol.connection_elements.ConnectionElement import ConnectionElement


class CREve(ConnectionElement):
    def __init__(self):
        self.base_p = Parameter("eve_base")
        self.eve_clone = QuantumRegister(1, "eve_clone")
        self.eve_measure = ClassicalRegister(1, "eve_measure")
        self.bases = None

    def init(self, n_bits: int, seed=None):
        random = np.random.RandomState(seed)
        self.bases = random.randint(2, size=n_bits)

    def qc(self, channel: QuantumRegister, i: int, ctx: dict) -> QuantumCircuit:
        eve_qc = QuantumCircuit(channel, self.eve_clone, self.eve_measure, name="CREve")
        eve_qc.ry(-self.base_p * 3.14 / 2, channel)

        eve_qc.measure(channel, self.eve_measure)

        eve_qc.x(self.eve_clone).c_if(self.eve_measure, 1)
        eve_qc.ry(-self.base_p * 3.14 / 2, self.eve_clone)
        eve_qc.swap(channel, self.eve_clone)

        if i is not None:
            eve_qc.assign_parameters({"eve_base": self.bases[i]}, inplace=True)
        return eve_qc

    def cregs(self) -> list:
        return [self.eve_measure]

    def qregs(self) -> list:
        return [self.eve_clone]

    def input_params(self) -> list:
        return [self.base_p]

    def input_values(self) -> list:
        return [self.bases]
