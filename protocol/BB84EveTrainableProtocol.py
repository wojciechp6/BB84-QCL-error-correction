from typing import List

import torch
from qiskit_machine_learning.neural_networks import SamplerQNN

from protocol.BB84TrainableProtocol import BB84TrainableProtocol
from protocol.connection_elements.Layer import Layer
from protocol.connection_elements.QCLEve import QCLEve
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement


class BB84EveTrainableProtocol(BB84TrainableProtocol):
    def __init__(self, n_bits=50, seed=None, f_value:float=0.8,
                 *, batch_size=64, learning_rate=0.1):
        self.eve = QCLEve()
        super().__init__(n_bits, [self.eve], seed, batch_size=batch_size, learning_rate=learning_rate)
        self.f_value = f_value

    def train(self):
        losses = []
        for inputs, target, mask in self.dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(target, mask, outputs)
            loss.backward()
            self.optimizer.step()
            losses.append(loss)
        return torch.stack(losses).mean()

    def loss(self, target, mask, outputs):
        alpha = 10
        bob_output = outputs[:, 0]
        eve_output = outputs[:, 2]
        Fab = (bob_output - target)[mask].abs().mean()
        Fae = (eve_output - target)[mask].abs().mean()
        loss = alpha * (Fab - self.f_value) ** 2 - Fae
        return loss

    def froze_elements(self, elements_to_froze:List[TrainableConnectionElement]):
        frozen_params_names = [p.name for e in elements_to_froze for p in e.trainable_parameters()]
        params = self.get_parameters()
        frozen_params = {k: v for k, v in params.items() if k in frozen_params_names}
        qc = self._qc.assign_parameters(frozen_params)
        trainable_params = [p for p in self._trainable_params if p.name not in frozen_params_names]
        return SamplerQNN(circuit=qc, sampler=self._sampler, input_params=self._input_params, weight_params=trainable_params)
