from qiskit_aer.noise import depolarizing_error

from protocol.BB84Protocol import BB84Protocol
from protocol.connection_elements.Noise import Noise

if __name__ == "__main__":
    print("=== BB84 bez Eve, tylko szum ===")
    pipeline_clean = BB84Protocol(n_bits=100, elements=[Noise(depolarizing_error(0.1, 1))])
    acc, qber = pipeline_clean.run()
    print(f"Accuracy: {acc:.2%}, QBER: {qber:.2%}")

    # print("\n=== BB84 z Eve i szumem ===")
    # pipeline_eve = BB84Pipeline(n_bits=2, elements=[NoisyChannel(p_error=0.5), NoisyChannel(p_error=0.1)])
    # acc, qber = pipeline_eve.run()
    # print(f"Accuracy: {acc:.2%}, QBER: {qber:.2%}")
