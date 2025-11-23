from numbers import Integral
from typing import Sequence, Callable, List, Dict, cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.passmanager import BasePassManager
from qiskit.primitives import BaseSampler, BaseSamplerV1, BaseSamplerV2, PrimitiveResult
from qiskit.result import QuasiDistribution
from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.gradients import BaseSamplerGradient
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.neural_networks.sampler_qnn import SparseArray
from sympy.stats.rv import probability


class MultiOutputSamplerQNN(SamplerQNN):
    def __init__(
        self,
        *,
        circuit: QuantumCircuit,
        sampler: BaseSampler | None = None,
        input_params: Sequence[Parameter] | None = None,
        weight_params: Sequence[Parameter] | None = None,
        sparse: bool = False,
        interpret: Callable[[int], int | tuple[int, ...]] | None = None,
        output_shape: int | tuple[int, ...] | None = None,
        gradient: BaseSamplerGradient | None = None,
        input_gradients: bool = False,
        pass_manager: BasePassManager | None = None,
    ):
        super().__init__(
            circuit=circuit,
            sampler=sampler,
            input_params=input_params,
            weight_params=weight_params,
            sparse=sparse,
            interpret=interpret,
            output_shape=output_shape,
            gradient=gradient,
            input_gradients=input_gradients,
            pass_manager=pass_manager,
        )

        self.num_outputs = len(circuit.cregs)

    def _postprocess(self, num_samples: int, result: PrimitiveResult) -> Dict[str, np.ndarray | SparseArray]:
        prob = {}

        if not isinstance(self.sampler, BaseSamplerV2):
            raise QiskitMachineLearningError(
                "The accepted estimator is BaseSamplerV2; "
                + f"got {type(self.sampler)} instead."
            )
        for i in range(num_samples):
            data = result[i].data
            for reg_name, reg_result in data.items():
                bitstring_counts = reg_result.get_counts()

                # Normalize the counts to probabilities
                total_shots = sum(bitstring_counts.values())
                probabilities = {k: v / total_shots for k, v in bitstring_counts.items()}

                # Convert to quasi-probabilities
                counts = QuasiDistribution(probabilities)
                counts = {k: v for k, v in counts.items() if int(k) < 2 ** reg_result.num_bits}

                probabilities = prob.get(reg_name)
                if probabilities is None:
                    probabilities = np.zeros((num_samples, 2 ** reg_result.num_bits))

                # evaluate probabilities
                for b, v in counts.items():
                    key = self._interpret(b)
                    if isinstance(key, Integral):
                        key = (cast(int, key),)
                    key = (i, *key)  # type: ignore
                    probabilities[key] += v
                prob[reg_name] = probabilities

        return prob

    def _validate_forward_output(
        self, output_data: Dict[str, np.ndarray], original_shape: tuple[int, ...]
    ) -> Dict[str, np.ndarray | SparseArray]:
        return output_data

    def backward(
        self, input_data: np.ndarray | None, weights: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Backward pass that returns gradients as concatenated arrays.

        Args:
            input_data: Input data
            weights: Weight parameters

        Returns:
            Tuple of (input_grad, weight_grad) where gradients are concatenated
            arrays from all registers in sorted order.
        """
        try:
            # Wywołaj parent class backward to get raw gradients
            input_grad, weight_grad = super().backward(input_data, weights)
        except (TypeError, AttributeError) as e:
            # Fallback jeśli super().backward() nie istnieje
            print(f"Warning: backward() not available, returning None: {e}")
            return None, None

        # DEBUG: sprawdzenie struktur
        import os
        debug = os.environ.get("QML_DEBUG_GRADS", "") == "1"
        if debug:
            print(f"DEBUG MultiOutputSamplerQNN: input_grad type: {type(input_grad)}")
            print(f"DEBUG MultiOutputSamplerQNN: weight_grad type: {type(weight_grad)}")
            if isinstance(weight_grad, dict):
                for k, v in weight_grad.items():
                    print(f"  {k}: shape={v.shape}, norm={np.linalg.norm(v)}")

        # Jeśli parent zwrócił słownik, łączymy gradienty z wszystkich rejestrów
        if isinstance(input_grad, dict):
            # Pobierz nazwy rejestrów w sortowanej kolejności dla spójności
            register_names = sorted(input_grad.keys())
            input_grad_list = [input_grad[key] for key in register_names]
            # Łączymy wzdłuż osi output (axis=-2)
            input_grad = np.concatenate(input_grad_list, axis=-2) if input_grad_list else None

        if isinstance(weight_grad, dict):
            # Pobierz nazwy rejestrów w sortowanej kolejności dla spójności
            register_names = sorted(weight_grad.keys())
            weight_grad_list = [weight_grad[key] for key in register_names]
            if debug:
                for i, (k, w) in enumerate(zip(register_names, weight_grad_list)):
                    print(f"  concatenating {k}: shape={w.shape}, norm={np.linalg.norm(w)}")
            # Łączymy wzdłuż osi output (axis=-2)
            weight_grad = np.concatenate(weight_grad_list, axis=-2) if weight_grad_list else None
            if debug:
                print(f"  result shape: {weight_grad.shape}, norm: {np.linalg.norm(weight_grad)}")

        return input_grad, weight_grad
