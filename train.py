from typing import OrderedDict

import torch
import torch.optim as optim
from qiskit_aer.noise import depolarizing_error, phase_damping_error, thermal_relaxation_error, pauli_error, amplitude_damping_error

from experiments.plt_pab_fae import plt_fab_fae
from protocol.BB84EveTrainableProtocol import BB84EveTrainableProtocol
from protocol.BB84Protocol import BB84Protocol
from protocol.BB84TrainableProtocol import BB84TrainableProtocol
from protocol.connection_elements.Layer import SeparableLayer
from protocol.connection_elements.Noise import Noise
from protocol.connection_elements.QCLEve import QCLEve

if __name__ == "__main__":
    print("Trained pipeline")
    target_fab = 0.85
    pipeline_train = BB84EveTrainableProtocol(n_bits=1024, elements=[QCLEve()], f_value=target_fab,
                                              seed=0, learning_rate=0.1, batch_size=1024, torch_device="cpu", backend_device="CPU")

    qc, _ = pipeline_train.qc_with_ctx()
    qc.draw("mpl", filename="qc_full.png")
    qc.decompose().draw("mpl", filename="qc_full1.png")
    qc.decompose().decompose().draw("mpl", filename="qc_full2.png")
    qc, _ = pipeline_train.qc_with_ctx(1)
    qc.draw("mpl", filename="qc.png")
    qc.decompose().draw("mpl", filename="qc1.png")
    qc.decompose().decompose().draw("mpl", filename="qc2.png")

    print(f"Start parameters {pipeline_train.get_all_parameters()}")
    qber = pipeline_train.run()
    print(f"Before training: QBER: {qber}")

    bob_qbers, eve_qbers = [], []

    for epoch in range(50):
        loss = pipeline_train.train()
        print(f'epoch: {epoch}, loss: {loss}')
        qber = pipeline_train.run()
        print(f"training epoch {epoch}: {qber}")
        bob_qbers.append(qber['bob_qber'])
        eve_qbers.append(qber['eve_qber'])

    print(f"Final parameters {pipeline_train.get_all_parameters()}")
    print(f"After training QBER: {qber}")

    sd = pipeline_train.model.state_dict()
    sd = OrderedDict({k: torch.zeros_like(v) for k, v in sd.items()})
    pipeline_train.model.load_state_dict(sd)
    qber = pipeline_train.run()
    print(f"Zeroed parameters: QBER: {qber}")

    plt_fab_fae(bob_qbers, eve_qbers, target_fab).savefig("fab_fae.png")