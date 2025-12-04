import itertools
from typing import List

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer.primitives import EstimatorV2
from qiskit_machine_learning.optimizers.optimizer_utils import learning_rate
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from protocol.BB84Protocol import BB84Protocol
from protocol.connection_elements.ConnectionElement import ConnectionElement
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement
from qiskit_extension.MultiOutputEstimatorQNNWraper import MultiOutputEstimatorQNNWrapper
from qiskit_extension.MultiOutputQNNWraper import MultiOutputQNNWrapper


class BB84TrainableProtocol(BB84Protocol):
    def __init__(self, n_bits=50, elements:List[ConnectionElement]=None, seed:int=None,
                 *, batch_size:int=64, learning_rate:float=0.1, torch_device:str='cpu', backend_device:str='CPU'):
        super().__init__(n_bits, elements, seed, device=backend_device)

        self._trainable_params = [e.trainable_parameters() for e in self.elements if isinstance(e, TrainableConnectionElement)]
        self._trainable_params = list(itertools.chain.from_iterable(self._trainable_params))
        self._device = torch.device(torch_device)
        self._frozen_params = {}
        self._learning_rate = learning_rate
        self._estimator = self._get_estimator(self._ctx, backend_device)
        self.model = MultiOutputEstimatorQNNWrapper(self._qc, self._input_params, self._trainable_params,
                                                    device=self._device, estimator=self._estimator)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.dataloader = self._get_dataloader(batch_size)

    def _get_estimator(self, ctx, device) -> EstimatorV2:
        return EstimatorV2(options={"backend_options": {'noise_model': ctx['noise_model'], "device": device}})

    def _get_dataloader(self, batch_size) -> DataLoader:
        inputs = torch.stack([torch.tensor(v, dtype=torch.int) for v in self._input_values], dim=1).to(self._device)
        target = torch.tensor(self.alice.bits, dtype=torch.int, device=self._device)
        mask = (torch.tensor(self.alice.bases) == torch.tensor(self.bob.bases)).to(self._device)
        return DataLoader(TensorDataset(inputs, target, mask), batch_size, shuffle=True)

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
        params = self.get_all_parameters()
        qc = self._qc.assign_parameters(params)
        return self._run_and_calculate_qber(qc)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def get_unfrozen_parameters(self):
        param_values = next(self.model.parameters()).detach().cpu().numpy()
        params = {p.name: v for p, v in zip(self.model.qnn.weight_params, param_values)}
        return params

    def get_frozen_params(self):
        return self._frozen_params

    def get_all_parameters(self):
        params = self._frozen_params.copy()
        params.update(self.get_unfrozen_parameters())
        return params

    def freeze_elements(self, elements_to_froze:List[TrainableConnectionElement]):
        to_freeze_params_names = [p.name for e in elements_to_froze for p in e.trainable_parameters()]
        params = self.get_all_parameters()
        to_freeze_params = {k: v for k, v in params.items() if k in to_freeze_params_names}

        qc = self._qc.assign_parameters(to_freeze_params)
        trainable_params = [p for p in self._trainable_params if p.name not in to_freeze_params_names]
        weights = torch.tensor([v for k, v in params.items() if k not in to_freeze_params_names])

        self._frozen_params = to_freeze_params
        self.model = MultiOutputEstimatorQNNWrapper(qc, self._input_params, trainable_params,
                                                    device=self._device, estimator=self._estimator, initial_weights=weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self._learning_rate)

    def defrost_all_elements(self):
        params = self.get_all_parameters()
        weights = torch.tensor(list(params.values()))
        self._frozen_params = {}
        self.model = MultiOutputEstimatorQNNWrapper(self._qc, self._input_params, self._trainable_params,
                                                    device=self._device, estimator=self._estimator,
                                                    initial_weights=weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self._learning_rate)




