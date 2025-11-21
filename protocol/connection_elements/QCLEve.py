import math
from typing import List

from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator

from protocol.connection_elements.ConnectionElement import ConnectionElement
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement


class QCLEve(TrainableConnectionElement):
    def __init__(self):
        self.params = [Parameter(f"Λ{index}") for index in range(3)]
        self.ansatzs = [QLCAnsatz(f"eve_ansatz_{i}") for i in range(2)]

    def process(self, qc, i: int, ctx: dict):
        n = 1
        target, clone = 0, 1
        blk = QuantumRegister(n, "blank")
        qc.add_register(blk)
        for i in range(n):
            for ansatz in self.ansatzs:
                qc = ansatz.process(qc, i, ctx)
        qc.rx(self.params[0], clone)
        qc.ry(self.params[1], clone)
        qc.rz(self.params[2], clone)
        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return self.params + [param for ansatz in self.ansatzs for param in ansatz.trainable_parameters()]

class QLCAnsatz(TrainableConnectionElement):
    def __init__(self, name: str):
        super().__init__()
        self.params = [Parameter(f"{name}_{i}") for i in range(6)]

    def process(self, qc: QuantumCircuit, i: int, ctx: dict):
        qc.rx(self.params[0], qubit=0)
        qc.ry(self.params[1], qubit=0)
        qc.rz(self.params[2], qubit=0)
        qc.rx(self.params[3], qubit=1)
        qc.ry(self.params[4], qubit=1)
        qc.rz(self.params[5], qubit=1)
        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return self.params