from typing import OrderedDict

import torch
import torch.optim as optim
from qiskit_aer.noise import depolarizing_error, phase_damping_error, thermal_relaxation_error, pauli_error, amplitude_damping_error

from protocol.BB84Protocol import BB84Protocol
from protocol.BB84TrainableProtocol import BB84TrainableProtocol
from protocol.connection_elements.Layer import Layer
from protocol.connection_elements.Noise import Noise


if __name__ == "__main__":
    # channel_noise = Noise(pauli_error([('X', 0.5), ('I', 0.5)]))
    channel_noise = Noise(amplitude_damping_error(0.5))

    print("Classic pipeline")
    pipeline = BB84Protocol(n_bits=1000, elements=[channel_noise], seed=0)
    acc, qber = pipeline.run()
    print(f"Accuracy: {acc} QBER: {qber}")

    print("Trained pipeline")
    pipeline_train = BB84TrainableProtocol(n_bits=1000, elements=[Layer(), channel_noise, Layer()], seed=0,
                                           learning_rate=0.01, batch_size=64)
    print(f"Start parameters {pipeline_train.get_parameters()}")

    acc, qber = pipeline_train.run()
    print(f"Before training: Accuracy: {acc} QBER: {qber}")

    for e in range(0,4):
        for epoch in range(21):
            loss = pipeline_train.train()
            if epoch % 5 == 0:
                print(f'epoch: {epoch+e*20}, loss: {loss}')
        acc, qber = pipeline_train.run()
        print(f"training epoch {(e+1)*20}: Accuracy: {acc} QBER: {qber}")

    print(f"Final parameters {pipeline_train.get_parameters()}")
    print(f"After training: Accuracy: {acc} QBER: {qber}")

    sd = pipeline_train.model.state_dict()
    sd = OrderedDict({k: torch.zeros_like(v) for k, v in sd.items()})
    pipeline_train.model.load_state_dict(sd)
    acc, qber = pipeline_train.run()
    print(f"Zeroed parameters: Accuracy: {acc} QBER: {qber}")