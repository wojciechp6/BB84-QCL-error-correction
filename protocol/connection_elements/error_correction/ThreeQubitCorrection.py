from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from protocol.connection_elements.ConnectionElement import ConnectionElement


class ThreeQubitBitFlipEncoder(ConnectionElement):
    def __init__(self, name="3QubitBitFlipEncoder"):
        super().__init__()
        self.name = name

    def init(self, n_bits: int, channel_size:int=1, seed=None):
        assert channel_size == 3, "3 Qubit Error Correction requires a channel size of 3"

    def qc(self, channel: QuantumRegister, i: int|None, ctx: dict) -> QuantumCircuit:
        qc = QuantumCircuit(channel, name=self.name)
        qc.cx(channel[0], channel[1])
        qc.cx(channel[0], channel[2])
        return qc


class ThreeQubitBitFlipDecoder(ConnectionElement):
    def __init__(self, name="3QubitBitFlipDecoder"):
        super().__init__()
        self.name = name
        self.anscilla = QuantumRegister(2, name="anscilla")
        self.test = ClassicalRegister(2, name="testr")

    def init(self, n_bits: int, channel_size:int=1, seed=None):
        assert channel_size == 3, "3 Qubit Error Correction requires a channel size of 3"

    def qc(self, channel: QuantumRegister, i: int|None, ctx: dict) -> QuantumCircuit:
        qc = QuantumCircuit(channel, self.anscilla, self.test, name=self.name)
        qc.cx(channel[0], self.anscilla[0])
        qc.cx(channel[1], self.anscilla[0])
        qc.cx(channel[0], self.anscilla[1])
        qc.cx(channel[2], self.anscilla[1])

        qc.ccx(self.anscilla[0], self.anscilla[1], channel[0])
        qc.x(self.anscilla[1])
        qc.ccx(self.anscilla[0], self.anscilla[1], channel[1])
        qc.x(self.anscilla[0])
        qc.x(self.anscilla[1])
        qc.ccx(self.anscilla[0], self.anscilla[1], channel[2])
        qc.x(self.anscilla[0])
        # qc.measure(self.anscilla, self.test)

        qc.cx(channel[0], channel[2])
        qc.cx(channel[0], channel[1])
        return qc

    def qregs(self) -> list:
        return [self.anscilla]

    def cregs(self) -> list:
        return [self.test]

