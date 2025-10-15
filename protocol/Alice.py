import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


class Alice:
    def __init__(self, n_bits: int):
        self.bits = np.random.randint(2, size=n_bits)
        self.bases = np.random.randint(2, size=n_bits)
        self.bit_p = Parameter("alice_bit")
        self.base_p = Parameter("alice_base")

    def prepare(self, i=None) -> QuantumCircuit:
        qc = QuantumCircuit(1, 1)
        qc.rx(self.bit_p * 3.14, 0)
        qc.ry(self.base_p * 3.14/2, 0)
        if i is not None:
            qc.assign_parameters({"alice_base": self.bases[i], "alice_bit": self.bits[i]}, inplace=True)
        return qc
