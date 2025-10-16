from typing import OrderedDict

import torch
import torch.optim as optim
from qiskit_aer.noise import depolarizing_error, phase_damping_error, thermal_relaxation_error

from protocol.BB84Protocol import BB84Protocol
from protocol.BB84TrainableProtocol import BB84TrainableProtocol
from protocol.connection_elements.Layer import Layer
from protocol.connection_elements.Noise import Noise


if __name__ == "__main__":
    channel_noise = Noise(thermal_relaxation_error(0.2, 0.1, 0.01))
    encode = Layer()
    decode = Layer()

    print("Classic pipeline")
    pipeline = BB84Protocol(n_bits=1000, elements=[channel_noise])
    acc, qber = pipeline.run()
    print(f"Accuracy: {acc} QBER: {qber}")

    print("Trained pipeline")
    pipeline_train = BB84TrainableProtocol(n_bits=1000, elements=[encode, channel_noise, decode])
    sd = pipeline_train.model.state_dict()
    sd = OrderedDict({k: torch.zeros_like(v) for k, v in sd.items()})
    pipeline_train.model.load_state_dict(sd)

    acc, qber = pipeline_train.run()
    print(f"Accuracy: {acc} QBER: {qber}")

    for epoch in range(35):
        loss = pipeline_train.train()
        if epoch % 5 == 0:
            print(f'epoch: {epoch}, loss: {loss}')

    acc, qber = pipeline_train.run()
    print(f"Accuracy: {acc} QBER: {qber}")