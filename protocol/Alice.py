import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from protocol.connection_elements.ConnectionElement import ConnectionElement
from protocol.input_parameter import BoolInputParameter


class Alice(ConnectionElement):
    def __init__(self):
        self.bit_p = BoolInputParameter("alice_bit")
        self.base_p = BoolInputParameter("alice_base")

    def init(self, n_bits: int, channel_size=1, seed=None):
        random = np.random.RandomState(seed)
        self.bit_p.values = random.randint(2, size=n_bits)
        self.base_p.values = random.randint(2, size=n_bits)

    def input_params(self) -> list:
        return [self.bit_p, self.base_p]

    def qc(self, channel: QuantumRegister, i: int, ctx: dict) -> QuantumCircuit:
        ctx['alice_base_p'] = self.base_p
        qc = QuantumCircuit(channel, name="Alice")
        qc.rx(self.bit_p * math.pi, channel[0])
        qc.ry(self.base_p * math.pi/2, channel[0])
        if i is not None:
            qc.assign_parameters({"alice_base": self.base_p.values[i], "alice_bit": self.bit_p.values[i]}, inplace=True)
        return qc
