from typing import Dict, List, Tuple, Optional

import torch
from torch import nn, Tensor

from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


class MultiOutputEstimatorQNNWrapper(nn.Module):
    """
    Wrapper EstimatorQNN, który automatycznie wykrywa rejestry kwantowe
    i zwraca wartości <Z> dla każdej grupy qubitów w formie słownika:
        { reg_name: tensor(batch, len(reg)) }.

    Obwód NIE może mieć pomiarów.
    """

    def __init__(
        self,
        circuit,
        input_params,
        weight_params,
        *,
        initial_weights: Optional[Tensor] = None,
        device: torch.device = torch.device("cpu"),
        estimator=None,
        input_gradients: bool = True,
    ):
        super().__init__()

        circuit = circuit.remove_final_measurements(inplace=False)
        self.device = device

        # --- (1) Automatyczne wykrywanie rejestrów ---
        # circuit.qregs = [QuantumRegister('channel',1), QuantumRegister('eve',1), ...]
        self.reg_qubits: Dict[str, List[int]] = {}
        global_index = 0

        for qreg in circuit.qregs:
            n = len(qreg)
            indices = list(range(global_index, global_index + n))
            self.reg_qubits[qreg.name] = indices
            global_index += n

        # --- (2) Tworzymy obserwable Z dla wszystkich qubitów ---
        num_qubits = circuit.num_qubits
        observables: List[SparsePauliOp] = []
        self._reg_slices: Dict[str, Tuple[int, int]] = {}

        obs_index = 0
        for reg_name, qubits in self.reg_qubits.items():
            start = obs_index
            for q in qubits:
                # Qiskit: lewy znak to qubit N-1, prawy to qubit 0
                pauli = ["I"] * num_qubits
                pauli[num_qubits - 1 - q] = "Z"
                observables.append(SparsePauliOp("".join(pauli), coeffs=[1.0]))
                obs_index += 1
            end = obs_index
            self._reg_slices[reg_name] = (start, end)

        # --- (3) EstimatorQNN ---
        self.qnn = EstimatorQNN(
            circuit=circuit,
            observables=observables,
            input_params=input_params,
            weight_params=weight_params,
            input_gradients=input_gradients,
            estimator=estimator,   # opcjonalnie własny backend
        )

        # --- (4) TorchConnector ---
        self.base = TorchConnector(self.qnn, initial_weights).to(device)

    # wygodny dostęp do wag
    @property
    def weights(self) -> Tensor:
        return self.base._weights

    def parameters(self, recurse: bool = True):
        return self.base.parameters(recurse=recurse)

    # --- Główne API ---

    def forward(self, x: Tensor) -> Tensor:
        """
        Zwraca tensor (batch, liczba_obserwabli)
        kolejność zgodna z automatycznym wykryciem rejestrów.
        """
        return self.base(x.to(self.device))

    def forward_as_dict(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Zwraca słownik: { rejestr → <Z> na jego qubitach }
        Każda wartość ma shape (batch, |rejestr|).
        """
        out = self.forward(x)
        result = {}

        for reg_name, (start, end) in self._reg_slices.items():
            result[reg_name] = out[:, start:end]

        return result
