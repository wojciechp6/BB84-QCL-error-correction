from abc import abstractmethod, ABC
from typing import List

import torch
from qiskit.circuit import Parameter

from protocol.connection_elements.ConnectionElement import ConnectionElement

class TrainableConnectionElement(ConnectionElement, ABC):
    @abstractmethod
    def trainable_parameters(self) -> List[Parameter]:
        pass
