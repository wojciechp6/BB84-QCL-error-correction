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

        dy = torch.tensor(f_target - 0.5)
        dx_abs = -torch.sqrt(0.5 ** 2 - dy ** 2)
        theta = torch.arctan2(dx_abs, -dy)

        loss = (
                eve_f * torch.sin(theta) + bob_f * torch.cos(theta) +
                -2 * ((0.5 - bob_f) ** 2 + (0.5 - eve_f) ** 2 - 0.25)
                )
        return loss

    def loss2(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value

        loss = -0.5*(bob_f + eve_f) - torch.sqrt((0.5 - bob_f) ** 2 + (0.5 - eve_f) ** 2) + (bob_f - f_target).pow(2)
        return loss

    def loss3(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value

        loss = (-(bob_f + eve_f) + 5*(torch.abs((0.5 - bob_f) ** 2 + (0.5 - eve_f) ** 2 - 0.25))
                + 2*eve_f*(bob_f - f_target).abs())
        return loss

    def loss4(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value

        loss = (-(bob_f + eve_f) + (bob_f - f_target).abs())
        return loss

    def loss5(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value
        pi_target = torch.tensor(f_target * torch.pi)

        loss = ((bob_f * torch.cos(pi_target) - eve_f * torch.sin(pi_target))
                + 2 * torch.abs((0.5 - bob_f) ** 2 + (0.5 - eve_f) ** 2 - 0.25)
                + eve_f * (bob_f - f_target).abs())
        return loss

    def loss6(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value
        pi_target = torch.tensor(f_target * torch.pi)

        loss = ((bob_f * torch.cos(pi_target) - eve_f * torch.sin(pi_target))
                + 2 * torch.abs((0.5 - bob_f) ** 2 + (0.5 - eve_f) ** 2 - 0.25)
                + 2 * eve_f * (bob_f - f_target).abs())
        return loss


    def loss7(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value

        dy = torch.tensor(f_target - 0.5)
        dx_abs = -torch.sqrt(0.5 ** 2 - dy ** 2)
        theta = torch.arctan2(dx_abs, -dy)

        loss = (torch.abs((eve_f - 0.5) * torch.cos(theta) - (bob_f - 0.5) * torch.sin(theta)) +
                    eve_f * torch.sin(theta) + bob_f * torch.cos(theta) +
                    2 * torch.abs((0.5 - bob_f) ** 2 + (0.5 - eve_f) ** 2 - 0.25))
        return loss

    def loss8(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value

        dy = torch.tensor(f_target - 0.5)
        dx_abs = -torch.sqrt(0.5 ** 2 - dy ** 2)
        theta = torch.arctan2(dx_abs, -dy)

        loss = (
                torch.abs((eve_f - 0.5) * torch.cos(theta) - (bob_f - 0.5) * torch.sin(theta)) +
                    eve_f * torch.sin(theta) + bob_f * torch.cos(theta)
                    # 2 * torch.abs((0.5 - bob_f) ** 2 + (0.5 - eve_f) ** 2 - 0.25)
        )
        return loss

    def loss9(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value

        dy = torch.tensor(f_target - 0.5)
        dx_abs = -torch.sqrt(0.5 ** 2 - dy ** 2)
        theta = torch.arctan2(dx_abs, -dy)

        loss = (
                # torch.abs((eve_f - 0.5) * torch.cos(theta) - (bob_f - 0.5) * torch.sin(theta)) +
                    eve_f * torch.sin(theta) + bob_f * torch.cos(theta) +
                    2 * torch.abs((0.5 - bob_f) ** 2 + (0.5 - eve_f) ** 2 - 0.25)
        )
        return loss

    def loss10(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value

        dy = torch.tensor(f_target - 0.5)
        dx_abs = -torch.sqrt(0.5 ** 2 - dy ** 2)
        theta = torch.arctan2(dx_abs, -dy)

        loss = (
                # torch.abs((eve_f - 0.5) * torch.cos(theta) - (bob_f - 0.5) * torch.sin(theta)) +
                    eve_f * torch.sin(theta) + bob_f * torch.cos(theta) +
                    -2 * ((0.5 - bob_f) ** 2 + (0.5 - eve_f) ** 2 - 0.25)
        )
        return loss

    def loss11(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value

        dy = torch.tensor(f_target - 0.5)
        dx_abs = -torch.sqrt(0.5 ** 2 - dy ** 2)
        theta = torch.arctan2(dx_abs, -dy)

        loss = (
                    torch.abs((eve_f - 0.5) * torch.cos(theta) - (bob_f - 0.5) * torch.sin(theta)) +
                    eve_f * torch.sin(theta) + bob_f * torch.cos(theta) +
                    -2 * ((0.5 - bob_f) ** 2 + (0.5 - eve_f) ** 2 - 0.25)
        )
        return loss

    def loss12(self, target, mask, outputs):
        bob_Z = outputs["channel"][:, 0]
        eve_Z = outputs[self.eve.eve_clone.name][:, 0]

        sign = 1 - 2 * target.long()
        bob_f = 0.5 * (1 + sign * bob_Z)
        eve_f = 0.5 * (1 + sign * eve_Z)

        bob_f = bob_f[mask].mean()
        eve_f = eve_f[mask].mean()

        f_target = self.f_value

        dy = torch.tensor(f_target - 0.5)
        dx_abs = -torch.sqrt(0.5 ** 2 - dy ** 2)
        theta = torch.arctan2(dx_abs, -dy)

        loss = (
                    # torch.abs((eve_f - 0.5) * torch.cos(theta) - (bob_f - 0.5) * torch.sin(theta)) +
                    eve_f * torch.sin(theta) + bob_f * torch.cos(theta)
                    # -2 * ((0.5 - bob_f) ** 2 + (0.5 - eve_f) ** 2 - 0.25)
        )
        return loss