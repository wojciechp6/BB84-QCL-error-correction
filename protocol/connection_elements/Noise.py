from typing import List

from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer.noise import QuantumError, NoiseModel

from protocol.connection_elements.ConnectionElement import ConnectionElement


class Noise(ConnectionElement):
    def __init__(self, quantum_error: QuantumError, name: str="Noise"):
        self.error = quantum_error
        self.name = name

    def qc(self, channel: QuantumRegister, i: int, ctx: dict) -> QuantumCircuit:
        qc = QuantumCircuit(channel, name=self.name)
        for qbit in channel:
            qc.append(self.error.to_instruction(), [qbit])
        return qc
