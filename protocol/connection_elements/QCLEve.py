from typing import List

from qiskit import QuantumRegister, QuantumCircuit, transpile, ClassicalRegister
from qiskit.circuit import Parameter

from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement


class QCLEve(TrainableConnectionElement):
    def __init__(self):
        self.params = [Parameter(f"Λ{index}") for index in range(3)]
        self.ansatzs = [QLCAnsatz(f"eve_ansatz_{i}") for i in range(2)]
        self.clone = QuantumRegister(1, "eve_clone")
        self.measure = ClassicalRegister(1, "eve_measure")

    def qc(self, channel: QuantumRegister, i: int, ctx: dict):
        qc = QuantumCircuit(channel, self.clone, name="QCLEve")
        for ansatz in self.ansatzs:
            qc.append(ansatz.qc(channel, self.clone))
        qc.rx(self.params[0], self.clone)
        qc.ry(self.params[1], self.clone)
        qc.rz(self.params[2], self.clone)

        # qc.measure(self.clone, self.measure)
        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return self.params + [param for ansatz in self.ansatzs for param in ansatz.trainable_parameters()]

    def qregs(self) -> list:
        return [self.clone]

    def cregs(self) -> list:
        return [self.measure]


class QLCAnsatz:
    def __init__(self, name: str):
        super().__init__()
        self.params = [Parameter(f"{name}_{i}") for i in range(6)]

    def qc(self, channel: QuantumRegister, clone: QuantumRegister) -> QuantumCircuit:
        qc = QuantumCircuit(channel, name="QLCAnsatz")
        qc.rx(self.params[0], channel)
        qc.ry(self.params[1], channel)
        qc.rz(self.params[2], channel)
        qc.rx(self.params[3], clone)
        qc.ry(self.params[4], clone)
        qc.rz(self.params[5], clone)
        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return self.params