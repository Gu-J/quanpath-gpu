from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
import time

def run_sim(qc):
    simulator = AerSimulator(method='statevector')
    qc.save_statevector()
    compiled = transpile(qc, simulator, optimization_level=0)
    start = time.perf_counter()
    simulator.run(compiled).result()
    end = time.perf_counter()
    return end - start

num_qubits = 28  # QFT规模，8比特你可以调

# 生成两个QFT电路
qft1 = QFT(num_qubits, do_swaps=False)  # 不自动加swap，自己控制
qft2 = QFT(num_qubits, do_swaps=False)

qc = QuantumCircuit(num_qubits)

qc.compose(qft1, inplace=True)
# qc.swap(0, 1)  # 中间插入一个swap门
qc.compose(qft2, inplace=True)

sim_time = run_sim(qc)
print(f"QFT + SWAP + QFT 模拟耗时: {sim_time:.4f} 秒")
