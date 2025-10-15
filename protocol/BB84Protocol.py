from typing import List

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from protocol.Bob import Bob
from protocol.Alice import Alice
from protocol.connection_elements.ConnectionElement import ConnectionElement


class BB84Protocol:
    def __init__(self, n_bits=50, elements:List[ConnectionElement]=None):
        self.n_bits = n_bits
        self.alice = Alice(n_bits)
        self.bob = Bob(n_bits)
        self.elements = elements if elements is not None else []

        self.bob_results = []
        self.sifted_alice = []
        self.sifted_bob = []

        self.backend = AerSimulator()
        self.noise = NoiseModel()
        self.parameters = None

    def run(self):
        ctx = self._setup()

        for i in range(self.n_bits):
            qc = self.alice.prepare(i)

            for elem in self.elements:
                qc = elem.process(qc, i, ctx)

            result = self.bob.measure(qc, i, ctx)
            self.bob_results.append(result)

        self._sift()
        return self._metrics()

    def _setup(self) -> dict:
        ctx = {'backend': self.backend, 'noise_model': self.noise}
        for elem in self.elements:
            elem.setup(ctx)
        return ctx

    def _sift(self):
        for i in range(self.n_bits):
            if self.alice.bases[i] == self.bob.bases[i]:
                self.sifted_alice.append(self.alice.bits[i])
                self.sifted_bob.append(self.bob_results[i])

    def _metrics(self):
        if len(self.sifted_alice) == 0:
            return None, None
        a = np.array(self.sifted_alice)
        b = np.array(self.sifted_bob)
        acc = np.sum(a == b) / len(a)
        qber = 1 - acc
        return acc, qber
