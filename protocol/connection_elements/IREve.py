import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from protocol.connection_elements.BaseEve import BaseEve
from protocol.input_parameter import BoolInputParameter


class IREve(BaseEve):
    def __init__(self):
        super().__init__()
        self.base_p = BoolInputParameter("eve_base")

    def init(self, n_bits: int, channel_size:int=1, seed=None):
        assert channel_size == 1, "IREve only supports channel size of 1"
        super().init(n_bits, channel_size, seed)
        random = np.random.RandomState(seed)
        self.base_p.values = random.randint(2, size=n_bits)

    def qc(self, channel: QuantumRegister, i: int, ctx: dict) -> QuantumCircuit:
        eve_qc = QuantumCircuit(channel, self.eve_clone, self.eve_measure, name="IREve")
        eve_qc.ry(-self.base_p * math.pi / 2, channel)

        eve_qc.measure(channel, self.eve_measure)

        eve_qc.x(self.eve_clone).c_if(self.eve_measure, 1)
        eve_qc.ry(self.base_p * math.pi / 2, self.eve_clone)
        eve_qc.swap(channel, self.eve_clone)

        if i is not None:
            eve_qc.assign_parameters({"eve_base": self.base_p.values[i]}, inplace=True)
        return eve_qc

    def input_params(self) -> list:
        return [self.base_p]
