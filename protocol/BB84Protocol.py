from typing import List

import numpy as np
from qiskit_aer.noise import NoiseModel

from protocol.Bob import Bob
from protocol.Alice import Alice
from protocol.connection_elements.ConnectionElement import ConnectionElement


class BB84Protocol:
    def __init__(self, n_bits=50, elements:List[ConnectionElement]=None, seed:int=None):
        self.n_bits = n_bits
        self.elements = elements if elements is not None else []
        self.alice = Alice(n_bits, seed=seed)
        self.bob = Bob(n_bits, seed=seed+1)

    def run(self):
        ctx = self._setup()
        bob_results = []

        for i in range(self.n_bits):
            qc = self.alice.prepare(i)

            for elem in self.elements:
                qc = elem.process(qc, i, ctx)

            result = self.bob.measure(qc, i, ctx)
            bob_results.append(result)

        sifted = self._sift(bob_results)
        return self._metrics(*sifted)

    def _setup(self) -> dict:
        ctx = {'noise_model': NoiseModel()}
        for elem in self.elements:
            elem.setup(ctx)
        return ctx

    def _sift(self, bob_results:List[int]):
        sifted_alice, sifted_bob = [], []
        for i in range(self.n_bits):
            if self.alice.bases[i] == self.bob.bases[i]:
                sifted_alice.append(self.alice.bits[i])
                sifted_bob.append(bob_results[i])
        return sifted_alice, sifted_bob

    def _metrics(self, sifted_alice:List[int], sifted_bob:List[int]):
        if len(sifted_alice) == 0:
            return None, None
        a = np.array(sifted_alice)
        b = np.array(sifted_bob)
        acc = np.sum(a == b) / len(a)
        qber = 1 - acc
        return acc, qber
