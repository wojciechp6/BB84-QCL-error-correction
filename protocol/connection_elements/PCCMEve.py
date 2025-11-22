import math

from qiskit import QuantumRegister, QuantumCircuit, transpile, ClassicalRegister
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator

from protocol.connection_elements.BaseEve import BaseEve
from protocol.connection_elements.ConnectionElement import ConnectionElement


class PCCMEve(BaseEve):
    def __init__(self, theta: float = math.pi / 2):
        super().__init__()
        self.pccm_block = self._phase_covariant_cloner_block(theta)

    def qc(self, channel: QuantumRegister, i: int, ctx: dict):
        alice_base_p = ctx["alice_base_p"]
        n = 1
        qc = QuantumCircuit(channel, self.eve_clone, self.eve_measure, name="PCCMEve")
        for i in range(n):
            qc.append(self.pccm_block, [channel[i], self.eve_clone])
        qc.ry(-alice_base_p * math.pi / 2, self.eve_clone)
        qc.measure(self.eve_clone, self.eve_measure)
        return qc

    @staticmethod
    def _phase_covariant_cloner_block(theta: float = math.pi / 2) -> QuantumCircuit:
        qc = QuantumCircuit(2, name="U_pc")
        target, clone = 0, 1

        qc.rx(math.pi / 2, target)
        qc.append(RYGate(theta).control(1), [target, clone])
        qc.append(RYGate(-math.pi).control(1), [clone, target])
        qc.rx(-math.pi / 2, target)
        qc.rx(-math.pi / 2, clone)
        return qc