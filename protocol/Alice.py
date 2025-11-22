import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from protocol.connection_elements.ConnectionElement import ConnectionElement


class Alice(ConnectionElement):
    def __init__(self):
        self.bits = None
        self.bases = None
        self.bit_p = Parameter("alice_bit")
        self.base_p = Parameter("alice_base")

    def init(self, n_bits: int, seed=None):
        random = np.random.RandomState(seed)
        self.bits = random.randint(2, size=n_bits)
        self.bases = random.randint(2, size=n_bits)

    def input_params(self) -> list:
        return [self.bit_p, self.base_p]

    def input_values(self) -> list:
        return [self.bits, self.bases]

    def qc(self, channel: QuantumRegister, i: int, ctx: dict) -> QuantumCircuit:
        ctx['alice_base_p'] = self.base_p
        ctx['alice_bases'] = self.bases
        qc = QuantumCircuit(channel, name="Alice")
        qc.rx(self.bit_p * math.pi, channel)
        qc.ry(self.base_p * math.pi/2, channel)
        if i is not None:
            qc.assign_parameters({"alice_base": self.bases[i], "alice_bit": self.bits[i]}, inplace=True)
        return qc
