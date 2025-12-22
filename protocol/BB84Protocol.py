from collections.abc import Iterable
from typing import List, Tuple

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.primitives import PrimitiveResult
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2

from protocol.Bob import Bob
from protocol.Alice import Alice
from protocol.connection_elements.BaseEve import BaseEve
from protocol.connection_elements.ConnectionElement import ConnectionElement


class BB84Protocol:
    def __init__(self, n_bits=50, elements:List[ConnectionElement]=None, channel_size:int=1, seed:int=None, *, device:str='CPU'):
        self.n_bits = n_bits
        self.alice = Alice()
        self.bob = Bob()
        self.elements = elements if elements is not None else []
        self.elements = [self.alice] + self.elements + [self.bob]
        self.eve = self._get_eve(self.elements)

        seed = seed if seed is not None else 0
        for i, elem in enumerate(self.elements):
            elem.init(n_bits, channel_size, seed+i)
        self._channel_size = channel_size
        self._qc, self._ctx = self.qc_with_ctx()
        self._qc = self._qc.decompose(reps=5)
        self._sampler = self._get_sampler(seed, self._ctx, device)
        self._input_params, self._input_values = self._get_inputs(self.elements)

    def _get_eve(self, elements:List[ConnectionElement]) -> BaseEve | None:
        for elem in elements:
            if isinstance(elem, BaseEve):
                return elem
        return None

    def _get_inputs(self, elements:List[ConnectionElement]) -> Tuple[List, List]:
        input_params = []
        input_values = []
        for elem in elements:
            input_params.extend(elem.input_params())
            input_values.extend(elem.input_values())

        assert len(input_params) == len(input_values)
        for vals in input_values:
            assert len(vals) == self.n_bits
        return input_params, input_values

    def qc_with_ctx(self, i: int = None) -> Tuple[QuantumCircuit, dict]:
        ctx = self._get_ctx()
        channel = QuantumRegister(self._channel_size, "channel")
        qc = QuantumCircuit(channel)
        for elem in self.elements:
            qc.add_register(*elem.regs())
            qubits = list(channel) + [qbit for qreg in elem.qregs() for qbit in qreg]
            cbits = [cbit for creg in elem.cregs() for cbit in creg]
            qc.append(elem.qc(channel, i, ctx), qubits, cbits)
        return qc, ctx

    def _get_sampler(self, seed, ctx, device) -> SamplerV2:
        return SamplerV2(default_shots=1, seed=seed,
                  options={"backend_options": {'noise_model': ctx['noise_model'], "device": device}})

    def run(self):
        return self._run_and_calculate_qber(self._qc)

    def _run_and_calculate_qber(self, qc: QuantumCircuit):
        has_eve = self.eve is not None
        registers = [self.bob.measure.name]
        if has_eve:
            registers.append(self.eve.eve_measure.name)
        results = self._run_qc(qc, registers)
        qbers = {"bob_qber": self._metrics(*self._sift(results[self.bob.measure.name]))}
        if has_eve:
            qbers["eve_qber"] = self._metrics(*self._sift(results[self.eve.eve_measure.name]))
        return qbers

    def _run_qc(self, qc: QuantumCircuit, registers=("bob_measure",)) -> dict:
        input = np.stack(self._input_values, axis=1)
        pubs = [(qc, {p.name: v for p, v in zip(self._input_params, input[i])}) for i in range(self.n_bits)]
        results = self._sampler.run(pubs).result()
        return {r: [int(self._first_result(results, i, r)) for i in range(self.n_bits)] for r in registers}

    @staticmethod
    def _first_result(results: PrimitiveResult, index: int, register_name: str = "bob_measure") -> str:
        return results[index].data[register_name].get_bitstrings(0)[0]

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
