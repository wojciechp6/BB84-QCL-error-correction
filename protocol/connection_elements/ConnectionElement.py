from qiskit import QuantumCircuit


class ConnectionElement:
    def setup(self, ctx: dict):
        """Opcjonalna metoda wywoływana raz na początku pipeline’u."""
        pass

    def process(self, qc: QuantumCircuit, i: int, ctx: dict) -> QuantumCircuit:
        """Metoda wywoływana dla każdego bitu."""
        raise NotImplementedError
