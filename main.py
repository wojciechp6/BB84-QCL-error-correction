import torch
from collections import OrderedDict
from qiskit_aer.noise import amplitude_damping_error

from protocol.BB84Protocol import BB84Protocol
from protocol.connection_elements.Noise import Noise
from protocol.BB84TrainableProtocol import BB84TrainableProtocol
from protocol.connection_elements.Layer import Layer
from protocol.connection_elements.PCCMEve import PCCMEve

if __name__ == "__main__":
    channel_noise = Noise(amplitude_damping_error(0.5))

    pipeline_train = BB84TrainableProtocol(n_bits=32, elements=[PCCMEve()], seed=0,
                                           learning_rate=0.1, batch_size=32)
    print("************ Train")
    loss = pipeline_train.train()
    print(f"loss: {loss}")
    print("++++++++++++ Run")
    qber = pipeline_train.run()
    print(f"QBER: {qber}")
    print("************ Train")
    loss = pipeline_train.train()
    print(f"loss: {loss}")
    print("++++++++++++ Run")
    qber = pipeline_train.run()
    print(f"QBER: {qber}")