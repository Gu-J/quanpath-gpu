from qiskit import QuantumCircuit,transpile
from circuits import *
import json
from trans import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--high", type=int, default=2)
parser.add_argument("--low", type=int, default=28)
parser.add_argument("--blocking", type=int, default=26)
parser.add_argument("--depth", type=int, default=100)
parser.add_argument("--circuit", type=str, default='Random')
parser.add_argument("--baseline", type=str, default='')
args = parser.parse_args()

args.nqubits= args.high+args.low
print(args.nqubits,'qubits')
print('blocking',args.blocking,', at least',2**(args.blocking-26)*2,'GB')
simulator = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)
simulator.set_options(
    blocking_enable=True,
    blocking_qubits=args.blocking,
)



if args.baseline!='':
    qc=generate_circuit(args,False)
    qc=transpile(qc, simulator, optimization_level=0)
    qc.save_statevector()
    start=time.time()
    job = simulator.run(qc)
    result = job.result()
    sv_data = result.get_statevector().data

    end = time.time()

    print(f'qiskit blocking: {end - start:.4f} 秒')

    del sv_data,result,job,qc




qc_list=trans(args)
qc=QuantumCircuit(args.nqubits)

first=1
for e in qc_list:
    if first:
        first=0
        first_qc=e
        continue
    qc=qc.compose(e)


def remove_unused_qubits(circuit):
    total_qubits = circuit.num_qubits
    used_qubits = set()

    # Step 1: 找出被使用的 qubit 索引
    for instr, qargs, _ in circuit.data:
        for q in qargs:
            used_qubits.add(circuit.qubits.index(q))

    # Step 2: 识别未使用的 qubit 索引
    all_qubits = set(range(total_qubits))
    unused_qubits = sorted(all_qubits - used_qubits)

    # Step 3: 如果没有未使用的 qubit，直接返回原电路
    if not unused_qubits:
        return circuit, []

    # Step 4: 创建新电路，仅保留被使用的 qubit
    used_qubits_sorted = sorted(used_qubits)
    new_circuit = QuantumCircuit(len(used_qubits_sorted), name=circuit.name)

    qubit_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_qubits_sorted)}

    for instr, qargs, cargs in circuit.data:
        new_qargs = [new_circuit.qubits[qubit_map[circuit.qubits.index(q)]] for q in qargs]
        new_circuit.append(instr, new_qargs, cargs)

    return new_circuit, unused_qubits
new_first_qc, deleted_qubits = remove_unused_qubits(first_qc)
print(f"Deleted qubit indices: {deleted_qubits}",',',new_first_qc.num_qubits,'qubits left')
new_first_qc.save_statevector()

# print(qc.depth())
# print(qc.draw(output='text', fold=-1))

qc=transpile(qc, simulator, optimization_level=0)
# print(qc.depth())
# print(qc.draw(output='text', fold=-1))

qc.save_statevector()
import time
start=time.time()
job = simulator.run(new_first_qc)
result = job.result()
sv_data = result.get_statevector().data
print(f'first: {time.time() - start:.4f} 秒')
job = simulator.run(qc)
result = job.result()
sv_data = result.get_statevector().data
end = time.time()
print(f'ours total: {end - start:.4f} 秒')

# print(sv_data)

