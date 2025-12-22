from typing import List

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter

from protocol.connection_elements.BaseEve import BaseEve
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement


class QCLEve(BaseEve, TrainableConnectionElement):
    def __init__(self, num_layers_U: int = 2):
        super().__init__()

        self.num_layers_U = num_layers_U
        self.u_layers = [QLCAnsatz(f"Θ{layer}") for layer in range(num_layers_U)]

        self.v_params_z = [Parameter(f"ΛZ_{k}") for k in range(3)]
        self.v_params_x = [Parameter(f"ΛX_{k}") for k in range(3)]

    def init(self, n_bits: int, channel_size:int=1, seed=None):
        assert channel_size==1, "This Eve supports only one channel"

    def qc(self, channel: QuantumRegister, i: int, ctx: dict):
        alice_base_p = ctx["alice_base_p"]
        alice_bases = ctx["alice_bases"]

        qc = QuantumCircuit(channel, self.eve_clone, self.eve_measure, name="QCLEve")
        for layer in self.u_layers:
            qc.append(layer.qc(channel, self.eve_clone), [channel, self.eve_clone])

        qc.barrier(self.eve_clone)

        qc.ry((1 - alice_base_p) * self.v_params_z[0], self.eve_clone)
        qc.rz((1 - alice_base_p) * self.v_params_z[1], self.eve_clone)
        qc.ry((1 - alice_base_p) * self.v_params_z[2], self.eve_clone)

        qc.ry(alice_base_p * self.v_params_x[0], self.eve_clone)
        qc.rz(alice_base_p * self.v_params_x[1], self.eve_clone)
        qc.ry(alice_base_p * self.v_params_x[2], self.eve_clone)

        qc.measure(self.eve_clone, self.eve_measure)

        if i is not None:
            qc.assign_parameters({"alice_base": alice_bases[i]}, inplace=True)
        return qc

    def trainable_parameters(self) -> List[Parameter]:
        u_params = [p for layer in self.u_layers for p in layer.trainable_parameters()]
        return u_params + self.v_params_z + self.v_params_x


class QLCAnsatz:
    def __init__(self, name: str):
        self.params = [Parameter(f"{name}_{i}") for i in range(6)]

    def qc(self, channel, clone):
        qc = QuantumCircuit(channel, clone, name="QLCAnsatz")

        qc.rx(self.params[0], channel)
        qc.ry(self.params[1], channel)
        qc.rz(self.params[2], channel)

        qc.rx(self.params[3], clone)
        qc.ry(self.params[4], clone)
        qc.rz(self.params[5], clone)

        qc.cx(channel, clone)

        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return self.params
