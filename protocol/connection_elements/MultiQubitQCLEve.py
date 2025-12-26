from typing import List

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector, Register

from protocol.connection_elements.BaseEve import BaseEve
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement
from protocol.connection_elements.U import NQubitU


class MultiQubitQCLEve(BaseEve, TrainableConnectionElement):
    def __init__(self):
        super().__init__()
        self.nu = None

        self.v_params_z = ParameterVector("ΛZ", 3)
        self.v_params_x = ParameterVector("ΛX", 3)

    def init(self, n_bits: int, channel_size:int=1, seed=None):
        super().init(n_bits, channel_size, seed)
        self.nu = NQubitU(channel_size*2)

    def qc(self, channel: QuantumRegister, i: int, ctx: dict):
        alice_base_p = ctx["alice_base_p"]
        alice_bases = ctx["alice_bases"]

        qc = QuantumCircuit(channel, self.eve_clone, self.eve_measure, name="MultiQubitQCLEve")
        concatenated = QuantumRegister(name="concatenated", bits=list(channel) + list(self.eve_clone))

        qc.append(self.nu.qc(concatenated), list(channel) + list(self.eve_clone), [])

        qc.barrier(self.eve_clone)

        qc.ry((1 - alice_base_p) * self.v_params_z[0], self.eve_clone[0])
        qc.rz((1 - alice_base_p) * self.v_params_z[1], self.eve_clone[0])
        qc.ry((1 - alice_base_p) * self.v_params_z[2], self.eve_clone[0])

        qc.ry(alice_base_p * self.v_params_x[0], self.eve_clone[0])
        qc.rz(alice_base_p * self.v_params_x[1], self.eve_clone[0])
        qc.ry(alice_base_p * self.v_params_x[2], self.eve_clone[0])

        qc.measure(self.eve_clone[0], self.eve_measure)

        if i is not None:
            qc.assign_parameters({"alice_base": alice_bases[i]}, inplace=True)
        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return self.nu.params() + self.v_params_z.params + self.v_params_x.params
