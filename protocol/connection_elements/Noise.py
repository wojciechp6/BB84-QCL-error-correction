from typing import List

from qiskit import QuantumCircuit
from qiskit_aer.noise import QuantumError

from protocol.connection_elements.ConnectionElement import ConnectionElement


class Noise(ConnectionElement):
    """
    Kanał tożsamościowy z szumem.
    Można wybrać typ szumu: depolarizing, phase.
    """

    _index = 0
    _gates = ["id"]
    _operation_map = {
        "id": lambda qc: qc.id(0),
    }
    def __init__(self, quantum_error: QuantumError | List[QuantumError]):
        self.gate = Noise._gates[Noise._index]
        self.error = quantum_error
        if not isinstance(self.error, list):
            self.error = [self.error]
        Noise._index = Noise._index + 1

    def setup(self, ctx: dict):
        if self.error is not None:
            for error in self.error:
                ctx['noise_model'].add_all_qubit_quantum_error(error, [self.gate], warnings=False)

    def process(self, qc: QuantumCircuit, i: int, ctx: dict) -> QuantumCircuit:
        Noise._operation_map[self.gate](qc)
        return qc
