from abc import abstractmethod, ABC
from typing import List

from qiskit.circuit import QuantumCircuit, QuantumRegister

from protocol.input_parameter import InputParameter


class ConnectionElement(ABC):
    def init(self, n_bits: int, channel_size:int=1, seed=None):
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

    def input_params(self) -> List[InputParameter]:
        return []