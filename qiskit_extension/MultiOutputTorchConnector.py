from typing import Dict, Tuple

import numpy as np
import torch
from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import NeuralNetwork

from torch import Tensor
from torch.autograd import Function


class _MultiOutputTorchNNFunction(Function):
    """Custom autograd function for multi-output QNN - returns tuple of tensors."""

    @staticmethod
    def forward(ctx, input_data: Tensor, weights: Tensor, neural_network: NeuralNetwork, sparse: bool, register_names_tuple, register_sizes_dict):
        if input_data.shape[-1] != neural_network.num_inputs:
            raise QiskitMachineLearningError(
                f"Invalid input dimension! Received {input_data.shape} and "
                + f"expected input compatible to {neural_network.num_inputs}"
            )

        ctx.neural_network = neural_network
        ctx.sparse = sparse
        ctx.register_names = list(register_names_tuple)
        ctx.register_sizes = register_sizes_dict
        ctx.save_for_backward(input_data, weights)

        result = neural_network.forward(
            input_data.detach().cpu().numpy(),
            weights.detach().cpu().numpy()
        )

        def map_to_tensor(result_array: np.ndarray, device) -> torch.Tensor:
            t = torch.as_tensor(result_array, dtype=torch.float, device=device)
            if len(input_data.shape) == 1 and len(t.shape) > 1:
                t = t[0]
            return t

        tensor_tuple = tuple(map_to_tensor(result[name], input_data.device) for name in ctx.register_names)
        return tensor_tuple

    @staticmethod
    def backward(ctx, *grad_outputs):
        import os
        debug = os.environ.get("QML_DEBUG_GRADS", "") == "1"

        if debug:
            print("DEBUG: backward called with grad_outputs:")
            for i, g in enumerate(grad_outputs):
                if g is not None:
                    print(f"  grad_outputs[{i}]: shape={g.shape}, norm={float(g.norm())}")
                else:
                    print(f"  grad_outputs[{i}]: None")

        input_data, weights = ctx.saved_tensors
        neural_network = ctx.neural_network

        if input_data.shape[-1] != neural_network.num_inputs:
            raise QiskitMachineLearningError(
                f"Invalid input dimension! Received {input_data.shape} and "
                + f"expected input compatible to {neural_network.num_inputs}"
            )

        # Pobierz jacobianów z neural_network
        input_grad_np, weights_grad_np = neural_network.backward(
            input_data.detach().cpu().numpy(),
            weights.detach().cpu().numpy()
        )

        if debug:
            print(f"DEBUG: weights_grad_np shape: {weights_grad_np.shape if weights_grad_np is not None else None}")
            print(f"DEBUG: input_grad_np shape: {input_grad_np.shape if input_grad_np is not None else None}")

        weights_grad = None
        input_grad = None

        # Przetwórz weights gradient
        if weights_grad_np is not None:
            w_np = np.asarray(weights_grad_np)  # shape: [batch, num_outputs, num_weights]
            w_t = torch.as_tensor(w_np, dtype=torch.float)

            weights_grad_list = []

            for i, g_out in enumerate(grad_outputs):
                if g_out is None:
                    continue

                if debug and i == 0:
                    print(f"DEBUG: Processing register {i}, grad shape: {g_out.shape}")

                # g_out shape: [batch, output_dim] - dla tego rejestru
                # g_out może mieć 1 lub więcej output wymiarów
                g = g_out.detach().cpu()  # shape: [batch, output_dim] lub [batch]

                # Dla każdego output wymaru w tym rejestrze
                if g.dim() == 1:
                    # Jeśli g ma wymiar [batch], znaczy że jest tylko 1 output dla tego rejestru
                    g = g.unsqueeze(-1)  # [batch] -> [batch, 1]

                # Teraz g ma wymiar [batch, output_dim]
                # w_t[:, i:i+g.shape[1], :] to jacobianów dla tego rejestru
                num_outputs_for_register = g.shape[1]
                w_i = w_t[:, i:i+num_outputs_for_register, :]  # [batch, output_dim, num_weights]

                if debug and i == 0:
                    print(f"DEBUG: w_i shape: {w_i.shape}, g shape: {g.shape}")
                    print(f"DEBUG: w_i norm: {w_i.norm()}, g norm: {g.norm()}")

                # g: [batch, output_dim] -> [batch, output_dim, 1]
                g_expanded = g.unsqueeze(-1)  # [batch, output_dim, 1]

                # Iloczyn element-wise: [batch, output_dim, num_weights] * [batch, output_dim, 1]
                # Potem sumujemy po batch i output_dim
                grad_w_i = torch.sum(w_i * g_expanded, dim=[0, 1])  # [num_weights]
                weights_grad_list.append(grad_w_i)

                if debug and i == 0:
                    print(f"DEBUG: grad_w_i norm: {grad_w_i.norm()}")

            if weights_grad_list:
                weights_grad = torch.sum(torch.stack(weights_grad_list), dim=0)
                if debug:
                    print(f"DEBUG: final weights_grad norm: {weights_grad.norm()}")
            else:
                weights_grad = torch.zeros_like(weights)

            weights_grad = weights_grad.to(weights.device)

        # Przetwórz input gradient
        if input_grad_np is not None:
            x_np = np.asarray(input_grad_np)  # shape: [batch, num_outputs, num_inputs]
            x_t = torch.as_tensor(x_np, dtype=torch.float)

            input_grad_list = []

            for i, g_out in enumerate(grad_outputs):
                if g_out is None:
                    continue

                g = g_out.detach().cpu()
                if g.dim() == 1:
                    g = g.unsqueeze(-1)  # [batch] -> [batch, 1]

                # x_t[:, i:i+g.shape[1], :] to jacobianów dla tego rejestru
                num_outputs_for_register = g.shape[1]
                x_i = x_t[:, i:i+num_outputs_for_register, :]  # [batch, output_dim, num_inputs]

                # g: [batch, output_dim] -> [batch, output_dim, 1]
                g_expanded = g.unsqueeze(-1)

                # Iloczyn i sumowanie: [batch, output_dim, num_inputs]
                grad_x_i = torch.sum(x_i * g_expanded, dim=[0, 1])  # [num_inputs]
                input_grad_list.append(grad_x_i)

            if input_grad_list:
                input_grad = torch.sum(torch.stack(input_grad_list), dim=0)
            else:
                input_grad = torch.zeros_like(input_data)

            input_grad = input_grad.to(input_data.device)

        return input_grad, weights_grad, None, None, None, None


class MultiOutputTorchConnector(TorchConnector):
    """Torch connector for multi-output QNN - zwraca słownik tensorów."""

    def __init__(self, neural_network: NeuralNetwork):
        """Inicjalizuje MultiOutputTorchConnector."""
        super().__init__(neural_network)
        if self._weights is not None:
            self._weights.requires_grad_(True)

        self._register_names = sorted([reg.name for reg in self._neural_network._circuit.cregs])
        self._register_sizes = {reg.name: 2 ** reg.size for reg in self._neural_network._circuit.cregs}

    def forward(self, input_data: Tensor | None = None) -> Tuple[Tensor, ...]:
        """Forward pass - zwraca tuple tensorów dla każdego rejestru."""
        input_ = input_data if input_data is not None else torch.zeros(0)

        output_tuple = _MultiOutputTorchNNFunction.apply(
            input_,
            self._weights,
            self._neural_network,
            self._sparse,
            tuple(self._register_names),
            self._register_sizes
        )

        return output_tuple

    def forward_as_dict(self, input_data: Tensor | None = None) -> Dict[str, Tensor]:
        """Forward pass - zwraca słownik tensorów dla każdego rejestru."""
        output_tuple = self.forward(input_data)
        result_dict = {name: tensor for name, tensor in zip(self._register_names, output_tuple)}
        return result_dict
