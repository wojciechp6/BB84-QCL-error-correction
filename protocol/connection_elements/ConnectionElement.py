from abc import abstractmethod, ABC

from qiskit import QuantumCircuit


class ConnectionElement(ABC):
    @abstractmethod
    def setup(self, ctx: dict):
        """Opcjonalna metoda wywoływana raz na początku pipeline’u."""
        pass

    @abstractmethod
    def process(self, qc: QuantumCircuit, i: int, ctx: dict) -> QuantumCircuit:
        """Metoda wywoływana dla każdego bitu."""
        pass
