import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator


class Bob:
    def __init__(self, n_bits: int):
        self.bases = np.random.randint(2, size=n_bits)
        self.base_p = Parameter("bob_base")
        self.backend = None

    def get_qc(self, qc: QuantumCircuit, i, ctx: dict):
        qc.ry(self.base_p * -3.14/2, 0)
        qc.measure(0, 0)
        if i is not None:
            qc.assign_parameters({"bob_base": self.bases[i]}, inplace=True)
        return qc

    def measure(self, qc: QuantumCircuit, i, ctx: dict) -> int:
        qc_bob = self.get_qc(qc, i, ctx)

        noise_model = ctx.get("noise_model", None)
        self.backend = AerSimulator(method="density_matrix", noise_model=ctx['noise_model'])
        compiled = transpile(qc_bob, self.backend, optimization_level=0)

        job = self.backend.run(compiled, shots=1, memory=True, noise_model=noise_model)
        return int(job.result().get_memory()[0])
