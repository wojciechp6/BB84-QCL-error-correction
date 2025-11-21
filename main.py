import torch
from collections import OrderedDict
from qiskit_aer.noise import amplitude_damping_error

from protocol.BB84Protocol import BB84Protocol
from protocol.connection_elements.CREve import CREve
from protocol.connection_elements.Noise import Noise
from protocol.BB84TrainableProtocol import BB84TrainableProtocol
from protocol.connection_elements.Layer import Layer
from protocol.connection_elements.PCCMEve import PCCMEve
from protocol.connection_elements.QCLEve import QCLEve

if __name__ == "__main__":
    # channel_noise = Noise(amplitude_damping_error(0.5))

    pipeline_train = BB84Protocol(n_bits=1000, elements=[CREve()], seed=0)
    print(pipeline_train.qc_with_ctx()[0])
    for epoch in range(20):
        # loss = pipeline_train.train()
        # print(f"loss: {loss}")
        qber = pipeline_train.run()
        print(f"QBER: {qber}")
