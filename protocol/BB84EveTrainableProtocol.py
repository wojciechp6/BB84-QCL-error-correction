import torch

from protocol.BB84TrainableProtocol import BB84TrainableProtocol


class BB84EveTrainableProtocol(BB84TrainableProtocol):
    def __init__(self, n_bits, elements, channel_size=1, seed=None, f_value:float=0.853, alpha=10,
                 *, batch_size=64, learning_rate:float=0.1, torch_device:str='cpu', backend_device:str='CPU'):
        super().__init__(n_bits, elements, channel_size, seed, batch_size=batch_size, learning_rate=learning_rate,
                         torch_device=torch_device, backend_device=backend_device)
        self.f_value = f_value
        self.alpha = alpha

    def loss(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value

        loss = self.alpha * (bob_f - f_target) ** 2 - eve_f
        return loss
