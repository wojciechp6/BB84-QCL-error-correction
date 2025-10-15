import torch.optim as optim
from qiskit_aer.noise import depolarizing_error, phase_damping_error

from protocol.BB84Protocol import BB84Protocol
from protocol.BB84TrainableProtocol import BB84TrainableProtocol
from protocol.connection_elements.Layer import Layer
from protocol.connection_elements.Noise import Noise


if __name__ == "__main__":
    channel_noise = Noise(phase_damping_error(0.5))
    encode = Layer()
    decode = Layer()
    elements = []
    pipeline_clean = BB84TrainableProtocol(n_bits=100, elements=[encode, channel_noise, decode])

    acc, qber = pipeline_clean.run()
    print(f"Accuracy: {acc} QBER: {qber}")

    for epoch in range(20):
        loss = pipeline_clean.train()
        print(f'epoch: {epoch}, loss: {loss}')
        print(pipeline_clean.get_params())

    acc, qber = pipeline_clean.run()
    print(f"Accuracy: {acc} QBER: {qber}")