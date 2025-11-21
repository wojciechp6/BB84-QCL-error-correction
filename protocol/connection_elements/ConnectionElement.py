from abc import abstractmethod, ABC

from qiskit import QuantumCircuit


class ConnectionElement(ABC):
    @abstractmethod
    def process(self, qc: QuantumCircuit, i: int, ctx: dict) -> QuantumCircuit:
        """Metoda wywoływana dla każdego bitu."""
        pass
