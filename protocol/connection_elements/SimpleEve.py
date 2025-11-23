from typing import List
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter

from protocol.connection_elements.BaseEve import BaseEve
from protocol.connection_elements.TrainableConnectionElement import TrainableConnectionElement


class SimpleEve(BaseEve, TrainableConnectionElement):
    """
    Bardzo prosty Eve do debugowania:
      1. Kopiuje kanał na qubit Ewy w bazie Z (CNOT channel -> eve_clone),
      2. Na klonie robi pojedynczą rotację RY(Θ_eve),
      3. Mierzy tylko qubit Ewy.
    To *celowo* niszczy stan kanału, więc Bob będzie miał duży QBER,
    ale za to gradienty po Θ_eve są bardzo wyraźne.
    """

    def __init__(self):
        super().__init__()
        self.theta = Parameter("Θ_eve")

    def qc(self, channel: QuantumRegister, i: int, ctx: dict):
        # i i ctx tutaj ignorujemy – chcemy maksymalnie prosty obwód
        qc = QuantumCircuit(channel, self.eve_clone, self.eve_measure, name="SimpleEve")

        # 1) Skopiuj bit Alice na klona w bazie Z
        qc.cx(channel, self.eve_clone)

        # 2) Parametryczna rotacja na klonie
        qc.ry(self.theta, self.eve_clone)

        # 3) Pomiar klona do rejestru klasycznego Ewy
        qc.measure(self.eve_clone, self.eve_measure)

        return qc

    def trainable_parameters(self) -> List[Parameter]:
        return [self.theta]

    def loss(self, input, target, mask, output):
        # Nie używana – loss liczysz w BB84EveTrainableProtocol
        pass
