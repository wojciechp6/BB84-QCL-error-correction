from typing import List

import torch
from qiskit_machine_learning.neural_networks import SamplerQNN
from sympy.physics.vector.printing import params

from protocol.BB84TrainableProtocol import BB84TrainableProtocol
from protocol.connection_elements.Layer import Layer
from protocol.connection_elements.QCLEve import QCLEve
from protocol.connection_elements.SimpleEve import SimpleEve
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement
from qiskit_extension.MultiOutputQNNWraper import MultiOutputQNNWrapper


class BB84EveTrainableProtocol(BB84TrainableProtocol):
    def __init__(self, n_bits, elements, seed=None, f_value:float=0.892, alpha=10,
                 *, batch_size=64, learning_rate:float=0.1, torch_device:str='cpu', backend_device:str='CPU'):
        super().__init__(n_bits, elements, seed, batch_size=batch_size, learning_rate=learning_rate,
                         torch_device=torch_device, backend_device=backend_device)
        self.f_value = f_value
        self.alpha = alpha

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

        f_target = self.f_value

        loss = self.alpha * (bob_f - f_target) ** 2 - eve_f
        return loss
