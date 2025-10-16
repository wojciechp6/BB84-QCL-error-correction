from typing import List

import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from protocol.connection_elements.ConnectionElement import ConnectionElement
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement


class Layer(TrainableConnectionElement):
    _index = 0
    def __init__(self):
        super().__init__()
        index = Layer._index
        Layer._index += 1
        self.params = [Parameter(f"x_{index}"), Parameter(f"y_{index}"), Parameter(f"z_{index}")]

    def setup(self, ctx: dict):
        pass

    def process(self, qc: QuantumCircuit, i: int, ctx: dict):
        qc.rx(self.params[0], qubit=0)
        qc.ry(self.params[1], qubit=0)
        qc.rz(self.params[2], qubit=0)
        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return self.params