from qiskit.circuit import Parameter
import numpy as np


class InputParameter(Parameter):
    def __init__(self, name: str, values=None, space=None):
        super().__init__(name)
        self.values = values
        self.space = space

    def cover_space(self, n_samples: int=8) -> np.ndarray:
        return np.random.choice(self.space, n_samples, replace=False)

class BoolInputParameter(InputParameter):
    def __init__(self, name, values=None):
        super().__init__(name, values, space=[0, 1])
        if values is not None:
            for v in values:
                if v not in [0, 1]:
                    raise ValueError("BoolInputParameter values must be boolean (0/1).")

    def cover_space(self, n_samples: int=2) -> np.ndarray:
        return np.array(self.space) if n_samples >=2 else np.random.choice(self.space)


