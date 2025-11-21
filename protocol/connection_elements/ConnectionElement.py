from abc import abstractmethod, ABC

from qiskit import QuantumCircuit, QuantumRegister


class ConnectionElement(ABC):
    def init(self, n_bits: int, seed=None):
        pass

    @abstractmethod
    def qc(self, channel: QuantumRegister, i: int|None, ctx: dict) -> QuantumCircuit:
        pass

    def regs(self) -> list:
        return self.cregs() + self.qregs()

    def cregs(self) -> list:
        return []

    def qregs(self) -> list:
        return []

    def input_params(self) -> list:
        return []

    def input_values(self) -> list:
        return []
