from typing import List

from qiskit import QuantumRegister, QuantumCircuit, transpile, ClassicalRegister
from qiskit.circuit import Parameter

from protocol.connection_elements.BaseEve import BaseEve
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement


class QCLEve(BaseEve, TrainableConnectionElement):
    def __init__(self):
        super().__init__()
        self.params = [Parameter(f"Λ{index}") for index in range(6)]
        self.ansatzs = [QLCAnsatz(f"Θ{i}") for i in range(2)]

    def qc(self, channel: QuantumRegister, i: int, ctx: dict):
        alice_base_p = ctx["alice_base_p"]
        alice_bases = ctx["alice_bases"]
        qc = QuantumCircuit(channel, self.eve_clone, self.eve_measure, name="QCLEve")
        for ansatz in self.ansatzs:
            qc.append(ansatz.qc(channel, self.eve_clone), [channel, self.eve_clone])

        qc.barrier(self.eve_clone)

        qc.rx(alice_base_p * self.params[0], self.eve_clone)
        qc.ry(alice_base_p * self.params[1], self.eve_clone)
        qc.rz(alice_base_p * self.params[2], self.eve_clone)

        qc.rx((1-alice_base_p) * self.params[3], self.eve_clone)
        qc.ry((1-alice_base_p) * self.params[4], self.eve_clone)
        qc.rz((1-alice_base_p) * self.params[5], self.eve_clone)

        qc.measure(self.eve_clone, self.eve_measure)

        if i is not None:
            qc.assign_parameters({"alice_base": alice_bases[i]}, inplace=True)
        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return self.params + [param for ansatz in self.ansatzs for param in ansatz.trainable_parameters()]

    def loss(self, input, target, mask, output):
        pass

class QLCAnsatz:
    def __init__(self, name: str):
        super().__init__()
        self.params = [Parameter(f"{name}_{i}") for i in range(6)]

    def qc(self, channel: QuantumRegister, clone: QuantumRegister) -> QuantumCircuit:
        qc = QuantumCircuit(channel, clone, name="QLCAnsatz")
        qc.rx(self.params[0], channel)
        qc.ry(self.params[1], channel)
        qc.rz(self.params[2], channel)
        qc.rx(self.params[3], clone)
        qc.ry(self.params[4], clone)
        qc.rz(self.params[5], clone)
        qc.cx(channel, clone)
        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return self.params