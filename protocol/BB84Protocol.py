from collections.abc import Iterable
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
        seed = seed if seed is not None else 0
        self.alice = Alice(n_bits, seed=seed)
        self.bob = Bob(n_bits, seed=seed+1)

    def run(self):
        ctx = self._get_ctx()
        bob_results = []

        for i in range(self.n_bits):
            qc = self.alice.prepare(i)

            for elem in self.elements:
                qc = elem.process(qc, i, ctx)

            result = self.bob.measure(qc, i, ctx)
            bob_results.append(result)

        sifted = self._sift(bob_results)
        return self._metrics(*sifted)

    def _get_ctx(self) -> dict:
        ctx = {'noise_model': NoiseModel()}
        return ctx

    def _sift(self, bob_results:Iterable[int]):
        mask = self.alice.bases == self.bob.bases
        sifted_alice = self.alice.bits[mask]
        sifted_bob = np.array(bob_results)[mask]
        return sifted_alice, sifted_bob

    def _metrics(self, sifted_alice:Iterable[int], sifted_bob:Iterable[int]):
        a = np.array(sifted_alice)
        b = np.array(sifted_bob)
        if len(b) == 0:
            return None, None
        acc = np.sum(a == b) / len(a)
        qber = 1 - acc
        return qber
