from typing import List

import torch
from qiskit_machine_learning.neural_networks import SamplerQNN

from protocol.BB84TrainableProtocol import BB84TrainableProtocol
from protocol.connection_elements.Layer import Layer
from protocol.connection_elements.QCLEve import QCLEve
from protocol.connection_elements.SimpleEve import SimpleEve
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement


class BB84EveTrainableProtocol(BB84TrainableProtocol):
    def __init__(self, n_bits, elements, seed=None, f_value:float=0.892,
                 *, batch_size=64, learning_rate:float=0.1):
        super().__init__(n_bits, elements, seed, batch_size=batch_size, learning_rate=learning_rate)
        self.f_value = f_value

    def train(self):
        losses = []
        for inputs, target, mask in self.dataloader:
            self.optimizer.zero_grad()
            outputs = self.model.forward_as_dict(inputs)
            loss = self.loss(target, mask, outputs)
            loss.backward()
            self.optimizer.step()
            losses.append(loss)
        return torch.stack(losses).mean()

    def loss(self, target, mask, outputs):
        bob_probs = outputs[self.bob.measure.name]  # (batch, 2)
        eve_probs = outputs[self.eve.eve_measure.name]  # (batch, 2)

        target_long = target.long()

        bob_p_correct = bob_probs.gather(1, target_long.unsqueeze(1)).squeeze(1)
        eve_p_correct = eve_probs.gather(1, target_long.unsqueeze(1)).squeeze(1)

        bob_f = bob_p_correct[mask].mean()
        eve_f = eve_p_correct[mask].mean()

        alpha = 10.0
        f_target = self.f_value

        loss = alpha * (bob_f - f_target) ** 2 - eve_f
        return loss

    def froze_elements(self, elements_to_froze:List[TrainableConnectionElement]):
        frozen_params_names = [p.name for e in elements_to_froze for p in e.trainable_parameters()]
        params = self.get_parameters()

        frozen_params = {k: v for k, v in params.items() if k in frozen_params_names}
        qc = self._qc.assign_parameters(frozen_params)
        trainable_params = [p for p in self._trainable_params if p.name not in frozen_params_names]
        return SamplerQNN(circuit=qc, sampler=self._sampler, input_params=self._input_params, weight_params=trainable_params)
