import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

from protocol.connection_elements import ConnectionElement


class Eve(ConnectionElement):
    def __init__(self):
        self.backend = Aer.get_backend("aer_simulator")

    def process(self, qc: QuantumCircuit, i: int, ctx: dict) -> QuantumCircuit:
        e_basis = np.random.randint(2)

        eve_qc = qc.copy()
        if e_basis == 1:
            eve_qc.h(0)
        eve_qc.measure(0, 0)

        compiled = transpile(eve_qc, self.backend)
        job = self.backend.run(compiled, shots=1, memory=True)
        measured_bit = int(job.result().get_memory()[0])

        resend_qc = QuantumCircuit(1, 1)
        if measured_bit == 1:
            resend_qc.x(0)
        if e_basis == 1:
            resend_qc.h(0)

        return resend_qc
