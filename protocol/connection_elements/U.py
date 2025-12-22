from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector, Qubit


class NQubitU:
    def __init__(self, channel_size: int):
        super().__init__()
        assert channel_size >= 2, "NQubitU requires at least 2 qubits to entangle."
        self.channel_size = channel_size
        self.uu_layers = self.create_uu_layers(channel_size)

    def qc(self, channel: QuantumRegister):
        assert self.channel_size == channel.size
        qc = QuantumCircuit(channel, name="NQ")

        for layer_i, uu_layer in enumerate(self.uu_layers):
            for uu_index, uu in enumerate(uu_layer):
                q1_index = uu_index * 2 + (layer_i % 2)
                q2_index = q1_index + 1
                qc.append(uu.qc(channel[q1_index], channel[q2_index]), [channel[q1_index], channel[q2_index]])
        return qc

    def params(self):
        return [p for uu_layer in self.uu_layers for uu in uu_layer for p in uu.params()]

    def create_uu_layers(self, channel_size: int):
        layers = []
        if channel_size == 2:
            layers.append([SimplifiedUU("W")])
        elif channel_size > 2:
            same = channel_size % 2 == 1
            l = channel_size // 2
            k = l if same else l - 1
            for i in range(channel_size):
                if i % 2 == 0:
                    layers.append([SimplifiedUU(f"W_{i}_{j}") for j in range(l)])
                else:
                    layers.append([SimplifiedUU(f"W_{i}_{j}") for j in range(k)])
        return layers

class SimplifiedUU:
    def __init__(self, name: str):
        self._u_gates = [U(f"{name}_U{i}") for i in range(2)]
        self._weyl = Weyl(f"{name}_weyl")
        self._name = name

    def qc(self, q1, q2):
        qreg = QuantumRegister(bits=[q1, q2], name="q",)
        qc = QuantumCircuit(qreg, name="UU")
        qc.append(self._u_gates[0].qc(q1), [q1])
        qc.append(self._u_gates[1].qc(q2), [q2])
        qc.append(self._weyl.qc(q1, q2), [q1, q2])
        return qc

    def params(self):
        return self._u_gates[0].params() + self._u_gates[1].params() + self._weyl.params()


class Weyl:
    def __init__(self, name: str):
        self._params = ParameterVector(f"{name}", length=3)

    def qc(self, q1, q2):
        qreg = QuantumRegister(bits=[q1, q2], name="q",)
        qc = QuantumCircuit(qreg, name="Weyl")
        qc.cx(q1, q2)
        qc.rz(self._params[0], q1)
        qc.rz(self._params[1], q2)
        qc.cx(q2, q1)
        qc.rz(self._params[2], q2)
        qc.cx(q1, q2)
        return qc

    def params(self):
        return self._params.params


class U:
    def __init__(self, name: str="U"):
        self._name = name
        self._params = ParameterVector(f"{name}", length=3)

    def qc(self, q):
        if isinstance(q, Qubit):
            qreg = QuantumRegister(bits=[q], name="q")
        else:
            qreg = q
            assert qreg.size == 1, "U gate requires a single qubit."
        qc = QuantumCircuit(qreg, name=self._name)
        qc.u(self._params[0], self._params[1], self._params[2], q)
        return qc

    def params(self):
        return self._params.params