import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer.primitives import SamplerV2

from utils import most_common_value


class Bob:
    def __init__(self, n_bits: int, seed=None):
        random = np.random.RandomState(seed)
        self.bases = random.randint(2, size=n_bits)
        self.base_p = Parameter("bob_base")
        self.backend = None

    def get_qc(self, qc: QuantumCircuit, i, ctx: dict) -> QuantumCircuit:
        qc.ry(self.base_p * -3.14/2, 0)
        qc.measure(0, 0)
        if i is not None:
            qc.assign_parameters({"bob_base": self.bases[i]}, inplace=True)
        return qc

    def measure(self, qc: QuantumCircuit, i, ctx: dict) -> int:
        qc_bob = self.get_qc(qc, i, ctx)

        noise_model = ctx.get("noise_model", None)
        self.backend = ctx.get("backend", None) or SamplerV2(options={"backend_options": {"noise_model": noise_model}})

        result = self.backend.run([qc_bob], shots=1).result()
        return int(most_common_value(result, 0))
