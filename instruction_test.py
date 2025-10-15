import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# 1. Definicja własnej instrukcji
id_noise_1 = Instruction(name="id_noise_1", num_qubits=1, num_clbits=0, params=[])

# 2. Backend i rozszerzenie targetu
backend = AerSimulator(method="density_matrix")
target = backend.target
target.add_instruction(id_noise_1, {(0,): None})
backend_custom = AerSimulator(method="density_matrix", target=target)

# 3. NoiseModel przypisany do custom gate
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.3, 1), ["id_noise_1"])

print("NoiseModel:")
print(noise_model)

# 4. Obwód testowy
qc = QuantumCircuit(1, 1)
qc.x(0)                       # przygotowanie |1>
qc.h(0)                       # superpozycja
qc.append(id_noise_1, [0])    # nasza customowa bramka z szumem
qc.h(0)                       # powrót do computational basis
qc.measure(0, 0)

print("\nCircuit:")
print(qc.draw())

# 5. Transpile i uruchomienie
compiled = transpile(qc, backend_custom, optimization_level=0)
job = backend_custom.run(compiled, shots=2000, noise_model=noise_model)
result = job.result()
counts = result.get_counts()

print("\nWyniki pomiarów:")
print(counts)

# 6. Obliczenie QBER (dla testu)
p0 = counts.get('0', 0) / 2000
p1 = counts.get('1', 0) / 2000
print(f"\nP(0)={p0:.3f}, P(1)={p1:.3f}")
