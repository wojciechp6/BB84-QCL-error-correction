import itertools
from typing import List

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN

from protocol.BB84Protocol import BB84Protocol
from protocol.connection_elements.ConnectionElement import ConnectionElement
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement
from qiskit_extension.MultiOutputSamplerQNN import MultiOutputSamplerQNN
from qiskit_extension.MultiOutputTorchConnector import MultiOutputTorchConnector


class BB84TrainableProtocol(BB84Protocol):
    def __init__(self, n_bits=50, elements:List[ConnectionElement]=None, seed:int=None,
                 *, batch_size:int=64, learning_rate:float=0.1):
        super().__init__(n_bits, elements, seed)

        self._trainable_params = [e.trainable_parameters() for e in self.elements if isinstance(e, TrainableConnectionElement)]
        self._trainable_params = list(itertools.chain.from_iterable(self._trainable_params))
        self.qnn = MultiOutputSamplerQNN(circuit=self._qc,
                         sampler=self._sampler,
                         input_params=self._input_params,
                         weight_params=self._trainable_params,
                                         input_gradients=True)
        self.model = MultiOutputTorchConnector(self.qnn)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.dataloader = self._get_dataloader(batch_size)

    def _get_dataloader(self, batch_size) -> DataLoader:
        inputs = torch.stack([torch.tensor(v, dtype=torch.int) for v in self._input_values], dim=1)
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
        qc = self._qc.assign_parameters(params)
        return self._run_and_calculate_qber(qc)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def get_parameters(self):
        param_values = next(self.model.parameters()).detach().cpu().numpy()
        params = {p.name: v for p, v in zip(self._trainable_params, param_values)}
        return params




