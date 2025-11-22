import torch
from collections import OrderedDict
from qiskit_aer.noise import amplitude_damping_error

from protocol.BB84EveTrainableProtocol import BB84EveTrainableProtocol
from protocol.BB84Protocol import BB84Protocol
from protocol.connection_elements.IREve import IREve
from protocol.connection_elements.Noise import Noise
from protocol.BB84TrainableProtocol import BB84TrainableProtocol
from protocol.connection_elements.Layer import Layer
from protocol.connection_elements.PCCMEve import PCCMEve
from protocol.connection_elements.QCLEve import QCLEve

if __name__ == "__main__":
    pipeline_train = BB84Protocol(n_bits=100, seed=0, elements=[PCCMEve()])
    print(pipeline_train.qc_with_ctx()[0])
    qber = pipeline_train.run()
    print(f"QBER: {qber}")
