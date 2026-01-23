from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from protocol.connection_elements.ConnectionElement import ConnectionElement

class FiveQubitEncoder(ConnectionElement):
    """
    [[5,1,3]] encoder (one common Clifford encoder; not unique).
    Conventions:
      - data qubits are ordered [q0,q1,q2,q3,q4]
      - input logical state is on q0, others start in |0>
    """
    
    def init(self, n_bits: int, channel_size:int=1, seed=None):
        assert channel_size == 5, "5 Qubit Error Correction requires a channel size of 5"

    def qc(self, channel: QuantumRegister, i: int|None, ctx: dict) -> QuantumCircuit:
        qc = QuantumCircuit(channel, name="5QubitEncoder")

        qc.z(channel[0])

        for i in range(1, 5):
            qc.h(channel[i])

        for i in range(1, 5):
            qc.cx(channel[i], channel[0])

        qc.cz(channel[4], channel[0])
        for i in range(4, 1, -1):
            qc.cz(channel[i-1], channel[i])

        return qc


@dataclass(frozen=True)
class FiveQubitDecoder(ConnectionElement):
    """
    [[5,1,3]] decoder (inverse of common Clifford encoder; not unique).
    Conventions:
      - data qubits are ordered [q0,q1,q2,q3,q4]
      - output logical state is on q0
    """

    def init(self, n_bits: int, channel_size:int=1, seed=None):
        assert channel_size == 5, "5 Qubit Error Correction requires a channel size of 5"

    def qc(self, channel: QuantumRegister, i: int|None, ctx: dict) -> QuantumCircuit:
        qc = QuantumCircuit(channel, name="5QubitDecoder")

        qc.cz(channel[4], channel[0])
        for i in range(2, 5):
            qc.cz(channel[i-1], channel[i])

        for i in range(1, 5):
            qc.cx(channel[i], channel[0])

        for i in range(1, 5):
            qc.h(channel[i])

        qc.z(channel[0])
        return qc