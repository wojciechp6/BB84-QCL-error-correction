from typing import List

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector

from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement
from protocol.connection_elements.U import NQubitU, U


class SeparableLayer(TrainableConnectionElement):
    def __init__(self, name="SeparableLayer"):
        super().__init__()
        self.name = name
        self.u_gates = None

    def init(self, n_bits: int, channel_size: int = 1, seed=None):
        self.u_gates = [U(f"{self.name}_U{i}") for i in range(channel_size)]

    def qc(self, channel: QuantumRegister, i: int, ctx: dict):
        assert len(channel) == len(self.u_gates), f"Channel size and number of U gates must match. {len(channel)} != {len(self.u_gates)}"
        qc = QuantumCircuit(channel, name=self.name)
        for qi in range(channel.size):
            qc.append(self.u_gates[qi].qc(channel[qi]), [channel[qi]])
        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return [p for u1 in self.u_gates for p in u1.params()]


class EntangledLayer(TrainableConnectionElement):
    def __init__(self, name="EntangledLayer"):
        super().__init__()
        self.name = name
        self.nu = None

    def init(self, n_bits: int, channel_size: int = 1, seed=None):
        self.nu = NQubitU(channel_size, name=f"{self.name}_NU")

    def qc(self, channel: QuantumRegister, i: int, ctx: dict):
        qc = QuantumCircuit(channel, name=self.name)
        qc.append(self.nu.qc(channel), channel)
        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return self.nu.params()

