import math
from typing import Dict, List, Tuple
import torch
from torch import nn, Tensor
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector


class MultiOutputQNNWrapper(nn.Module):
    """
    Wrapper, który:
      1) scala wszystkie classical registers w jeden rejestr 'c',
      2) buduje SamplerQNN z jednym rejestrem (działają gradienty),
      3) odtwarza logiczne rejestry (bob/eve/etc) przez marginalizację.
    """

    def __init__(
        self,
        qc: QuantumCircuit,
        sampler,
        input_params,
        weight_params,
        *,
        initial_weights=None,
        device: torch.device=torch.device("cpu"),
    ):
        """
        qc – oryginalny obwód Z WIELOMA classical registers
        sampler – SamplerV2
        input_params, weight_params – parametry do QNN
        """
        super().__init__()

        # --- 1) SCALANIE REJESTRÓW ---
        merged_qc, logical_reg_map = self._merge_registers(qc)

        self.logical_reg_map = logical_reg_map   # np {"bob":[0], "eve":[1]}
        self.num_qubits = len(next(iter(logical_reg_map.values()))) if len(logical_reg_map) > 0 else 0
        self.num_bits_total = sum(len(v) for v in logical_reg_map.values())

        # --- 2) BUDOWANIE QNN ---
        self.qnn = SamplerQNN(
            circuit=merged_qc,
            sampler=sampler,
            input_params=input_params,
            weight_params=weight_params,
            input_gradients=True
        )

        # --- 3) Torch connector ---
        self.base = TorchConnector(self.qnn, initial_weights).to(device)

    # ================================================================
    # REJESTRY → 1x CREG
    # ================================================================
    def _merge_registers(self, qc: QuantumCircuit):
        """
        Tworzy NOWY obwód:
          - z tymi samymi qubitami,
          - z jednym classical register 'c',
          - z przemapowanymi pomiarami.
        Zwraca:
           new_qc, logical_register_map
        """

        # --- 1. ZBIERZ INFORMACJE O REJESTRACH ---
        old_cregs = qc.cregs
        total_bits = sum(creg.size for creg in old_cregs)

        # Mapa logicznych rejestrów: nazwa → indeksy w nowym c[]
        logical_reg_map = {}
        current_index = 0
        for creg in old_cregs:
            logical_reg_map[creg.name] = list(range(current_index, current_index + creg.size))
            current_index += creg.size

        # --- 2. NOWY OBWÓD: te same qubity, jeden classical register ---
        new_qc = QuantumCircuit(qc.qubits, ClassicalRegister(total_bits, "c"))

        # --- 3. MAPOWANIE STARYCH KLASYcznych bitów → nowy c[index] ---
        bit_map = {}
        idx = 0
        for creg in old_cregs:
            for bit in creg:
                bit_map[bit] = new_qc.clbits[idx]
                idx += 1

        # --- 4. SKOPIUJ OPERACJE ---
        for inst, qargs, cargs in qc.data:
            if inst.name == "measure":
                # measure q -> c[new_index]
                old_bit = cargs[0]
                new_bit = bit_map[old_bit]
                new_qc.append(inst, qargs, [new_bit])
            else:
                # zwykłe bramki — kopiujemy normalnie
                new_qc.append(inst, qargs, cargs)

        return new_qc, logical_reg_map

    # ================================================================
    # PROXY → parametry QNN (żeby optimizer działał jak wcześniej)
    # ================================================================
    @property
    def _weights(self):
        return self.base._weights

    def parameters(self, recurse: bool = True):
        return self.base.parameters(recurse=recurse)

    # ================================================================
    # FORWARD – wspólny rozkład P(bitstring)
    # ================================================================
    def forward(self, x: Tensor) -> Tensor:
        return self.base(x)

    # ================================================================
    # MULTI-OUTPUT: słownik rejestrów
    # ================================================================
    def forward_as_dict(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Zwraca:
            {
                "bob": (batch, 2**|bob|),
                "eve": (batch, 2**|eve|),
                ...
            }
        Gradienty działają w 100%.
        """
        probs = self.forward(x)  # (batch, 2**n)

        if probs.dim() == 1:
            probs = probs.unsqueeze(0)

        batch, dim = probs.shape
        n_qubits = int(math.log2(dim))

        probs_nd = probs.view(batch, *([2] * n_qubits))

        out = {}
        for reg_name, bit_indices in self.logical_reg_map.items():

            axes_to_keep = bit_indices
            axes_to_sum = [i for i in range(n_qubits) if i not in axes_to_keep]

            reg_probs = probs_nd
            for ax in sorted([1 + a for a in axes_to_sum], reverse=True):
                reg_probs = reg_probs.sum(dim=ax)

            reg_probs = reg_probs.view(batch, -1)
            out[reg_name] = reg_probs

        return out
