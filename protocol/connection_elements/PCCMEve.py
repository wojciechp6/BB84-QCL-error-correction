import math

from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator

from protocol.connection_elements.ConnectionElement import ConnectionElement


class PCCMEve(ConnectionElement):
    def __init__(self):
        self.pccm_block = self._phase_covariant_cloner_block()

    def process(self, qc, i: int, ctx: dict):
        n = 1
        blk = QuantumRegister(n, "blank")
        qc.add_register(blk)
        for i in range(n):
            qc = qc.compose(self.pccm_block, [qc.qubits[i], blk[i]])
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