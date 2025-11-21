import itertools
from typing import List, Tuple

import numpy as np
import torch
from qiskit import QuantumCircuit
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN

from protocol.BB84Protocol import BB84Protocol
from protocol.connection_elements.ConnectionElement import ConnectionElement
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement
from utils import most_common_value


class BB84TrainableProtocol(BB84Protocol):
    def __init__(self, n_bits=50, elements:List[ConnectionElement]=None, seed:int=None,
                 *, batch_size:int=64, learning_rate:float=0.1, ):
        super().__init__(n_bits, elements, seed)

        self.qc, self.ctx = self._get_qc_with_ctx()
        self.sampler = SamplerV2(default_shots=1, seed=seed, options={"backend_options": {"noise_model": self.ctx.get("noise_model", None)}})
        self._trainable_params = [e.trainable_parameters() for e in self.elements if isinstance(e, TrainableConnectionElement)]
        self._trainable_params = list(itertools.chain.from_iterable(self._trainable_params))
        qnn = SamplerQNN(circuit=self.qc,
                         sampler=self.sampler,
                         input_params=[self.alice.bit_p, self.alice.base_p, self.bob.base_p],
                         weight_params=self._trainable_params)
        self.model = TorchConnector(qnn)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.dataloader = self._get_dataloader(batch_size)
        self.sampler.run_classic = self.sampler.run
        self.sampler.run = self.sampler_run

    def sampler_run(self, pubs, *, shots: int | None = None):
        print(f" Input: {[pub[1] for pub in pubs]}")
        r = self.sampler.run_classic(pubs, shots=shots)
        print(f" Output: {[most_common_value(r.result(), i) for i in range(len(pubs))]}")
        return r

    def _get_ctx(self) -> dict:
        ctx = super()._get_ctx()
        ctx['backend'] = SamplerV2(options={"backend_options": {"noise_model": ctx.get("noise_model", None)}})
        return ctx

    def _get_qc_with_ctx(self) -> Tuple[QuantumCircuit, dict]:
        ctx = self._get_ctx()
        trainable_qc = self.alice.prepare()
        for e in self.elements:
            trainable_qc = e.process(trainable_qc, 0, ctx)
        trainable_qc = self.bob.get_qc(trainable_qc, None, ctx)
        return trainable_qc, ctx

    def _get_dataloader(self, batch_size) -> DataLoader:
        inputs = torch.stack([
            torch.tensor([bit, basis_a, basis_b], dtype=torch.int)
            for bit, basis_a, basis_b
            in zip(self.alice.bits, self.alice.bases, self.bob.bases)
        ])
        target = torch.tensor(self.alice.bits, dtype=torch.int)
        mask = (torch.tensor(self.alice.bases) == torch.tensor(self.bob.bases))

        return DataLoader(TensorDataset(inputs, target, mask), batch_size)

    def train(self):
        losses = []
        for inputs, target, mask in self.dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs = outputs[:, 1]
            loss = (outputs - target)[mask].abs().mean()
            loss.backward()
            self.optimizer.step()
            losses.append(loss)
        return torch.stack(losses).mean()

    def run(self):
        params = self.get_parameters()
        qc = self.qc.assign_parameters(params)
        input = np.stack([self.alice.bits, self.alice.bases, self.bob.bases], axis=1)

        results = self.sampler.run([(qc, input[i]) for i in range(self.n_bits)]).result()
        bob_results = [int(most_common_value(results, i)) for i in range(self.n_bits)]

        sifted = self._sift(bob_results)
        return self._metrics(*sifted)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def get_parameters(self):
        param_values = next(self.model.parameters()).detach().cpu().numpy()
        params = {p.name: v for p, v in zip(self._trainable_params, param_values)}
        return params




