import itertools
from itertools import islice
from typing import List

import numpy as np
import torch
from qiskit import QuantumCircuit
from torch import nn, optim
from qiskit_aer.primitives import Estimator, Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

from protocol.BB84Protocol import BB84Protocol
from protocol.connection_elements.ConnectionElement import ConnectionElement


class BB84TrainableProtocol(BB84Protocol):
    def __init__(self, n_bits=50, elements:List[ConnectionElement]=None):
        super().__init__(n_bits, elements)
        self.sampler = Sampler()
        trainable_qc = self._prepare_trainable_qc()
        train_params = [list(e.params) for e in self.elements if hasattr(e, "params")]
        train_params = list(itertools.chain.from_iterable(train_params))
        qnn = SamplerQNN(circuit=trainable_qc,
                         input_params=[self.alice.bit_p, self.alice.base_p, self.bob.base_p],
                         weight_params=train_params)
        self.model = TorchConnector(qnn)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)

    def _prepare_trainable_qc(self) -> QuantumCircuit:
        ctx = self._setup()
        trainable_qc = self.alice.prepare()
        for e in self.elements:
            trainable_qc = e.process(trainable_qc, 0, ctx)
        trainable_qc = self.bob.get_qc(trainable_qc, None, ctx)
        return trainable_qc

    def train(self):
        inputs = torch.stack([
            torch.tensor([bit, basis_a, basis_b], dtype=torch.float32)
            for bit, basis_a, basis_b
            in zip(self.alice.bits, self.alice.bases, self.bob.bases)
        ])
        target = torch.tensor(self.alice.bits, dtype=torch.float32).unsqueeze(-1)
        mask = (torch.tensor(self.alice.bases) == torch.tensor(self.bob.bases))

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        outputs = outputs[:, 1].unsqueeze(-1)
        loss = ((outputs - target).abs())[mask].mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self):
        ctx = self._setup()
        bob_results = []

        for i in range(self.n_bits):
            qc = self.alice.prepare(i)

            for elem in self.elements:
                qc = elem.process(qc, i, ctx)

            qc.assign_parameters(self.get_params(), inplace=True)
            result = self.bob.measure(qc, i, ctx)
            bob_results.append(result)

        sifted = self._sift(bob_results)
        return self._metrics(*sifted)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def get_params(self):
        params = next(self.model.parameters()).detach().cpu().numpy()
        params = [params[i:i+3] for i in range(0, len(params), 3)]
        params = {f'{base}_{i}': float(p) for i, param in enumerate(params) for base, p in zip(['x', 'y', 'z'], param)}
        return params




