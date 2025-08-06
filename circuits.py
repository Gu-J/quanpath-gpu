from qiskit import QuantumCircuit, transpile, qasm2
from qiskit_aer import AerSimulator
from qiskit.circuit.library import *
import os
import random
import socket
from argparse import ArgumentParser
import time
from math import pi, log2, sqrt
import numpy as np
from datetime import datetime
from qiskit.circuit.random import random_circuit

random.seed(42)

def get_timestamp():
    return datetime.now()


def get_options():
    argparser = ArgumentParser(description="Input arguments for distributed simulations")
    argparser.add_argument(
        "-q", "--qasm", type=str, default="", help="QASM file name"
    )
    argparser.add_argument(
        "-c", "--circuit", type=str,
        default="random",
        help='Specify the quantum circuit name'
    )
    argparser.add_argument(
        "-n", "--nqubits", type=int, default=10, help="Number of qubits (default 10)"
    )
    argparser.add_argument(
        "-d", "--depth", type=int, default=10, help="Circuit depth (default 10)"
    )
    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(default_output_dir, exist_ok=True)
    default_output_file = os.path.join(default_output_dir, "output.txt")
    argparser.add_argument(
        "-o", "--output", type=str, default=default_output_file, help="The output file name"
    )
    return argparser.parse_args()

def generate_circuit(args, dump=True):
    # generate a quantum circuit
    basis=["cx", "u"]
    if args.circuit == "random":
        qc = generate_circuit_with_layer_control(args.nqubits, args.depth)
    elif args.circuit == "Random":
        qc=random_circuit(num_qubits=args.nqubits, depth=args.depth, measure=False,seed=46,max_operands=2)
        basis=['cx','cy', 'cz', 'ccx', 'cswap','h','rx']
        qc=transpile(qc, basis_gates=basis, optimization_level=0)
    elif args.circuit == "Random_separable":
        qcl=random_circuit(num_qubits=args.low, depth=args.depth, measure=False,seed=46,max_operands=2)
        qch=random_circuit(num_qubits=args.high, depth=args.depth, measure=False,seed=46,max_operands=2)
        qc=QuantumCircuit(args.nqubits)
        qc.compose(qcl,qubits=range(args.low),inplace=True)
        qc.compose(qch,qubits=range(args.low,args.nqubits),inplace=True)
        qc=transpile(qc, basis_gates=basis, optimization_level=0)
    elif args.circuit == "qv":
        qc = QuantumVolume(args.nqubits, seed=42).decompose() # , args.depth
        qc = transpile(qc, basis_gates=basis, optimization_level=0)
    elif args.circuit == "qft":
        qc = QFT(args.nqubits).decompose()
        # def QFT_Quirk(num_qubits):
        #     qc = QuantumCircuit(num_qubits, name="QFT")
            
        #     # 第一个循环：对右半部分 qubit 加 H 门
        #     # for i in range(num_qubits - 1, num_qubits // 2 - 1, -1):
        #     #     qc.h(num_qubits - 1 - i)
        #     # qc.barrier()
        #     # # 第二个循环：经典的 QFT 构造
        #     for i in range(num_qubits - 1, -1, -1):
        #         target = num_qubits - 1 - i
        #         qc.h(target) 
        #         for j in range(i - 1, -1, -1):
        #             control = num_qubits - 1 - j
        #             k = i - j + 1
        #             angle = 2 * pi / (2 ** k)
        #             qc.cp(angle, target, control)  # cu1 → cp in Qiskit

        #     # # 第二个循环：经典的 QFT 构造
        #     # for i in range(num_qubits):
        #     #     qc.h(i) 
        #     #     for j in range(i + 1, num_qubits):
        #     #         control = i
        #     #         target=j
        #     #         angle =  pi / (2 ** (j-i))
        #     #         qc.cp(angle, control, target)  # cu1 → cp in Qiskit
        #     #         # if control==0 and target==num_qubits-3:
        #     #         #     qc.barrier()
            
        #     return qc
        # qc=QuantumCircuit(30)
        # qc.compose(QFT_Quirk(28),qubits=range(28),inplace=True)
        # qc.h(29)
        # qc.ch(29,28)
        # qc.h(28)
        # qc=QFT_Quirk(args.nqubits)

        # basis=['cx','rx','rz','x']
        qc = transpile(qc, basis_gates=basis, optimization_level=0)
    elif args.circuit == "vqc_aa":
        qc = VQC_AA(args.nqubits)
    elif args.circuit == "iqp":
        qc = myIQP(args.nqubits)
        qc = transpile(qc, basis_gates=basis, optimization_level=0)
    else:
        print(f"[ERROR] Unknown circuit: {args.circuit}")
        exit(1)
    if dump:
        dump_qasm(qc, args.qasm)
    print('#'*100,qc.depth())
    return qc

def generate_circuit_with_layer_control(num_qubits, num_depths, proportion=0.1):
    qc = QuantumCircuit(num_qubits)
    num_swaps = int(proportion * num_depths)
    layer_operations = [[('x', j, j) for j in range(num_qubits)] for _ in range(num_depths)]

    for i in range(num_swaps):
        layer = random.randint(0, num_depths - 1)
        ctrl, targ = random.sample(range(0, num_qubits), 2)
        layer_operations[layer][ctrl] = ('swap', ctrl, targ)
        layer_operations[layer][targ] = ('I', targ, targ)
    for i in range(num_depths):
        for gate_type, ctrl, targ in layer_operations[i]:
            if gate_type == 'swap':
                qc.swap(ctrl, targ)
            elif gate_type == 'x':
                # qc.x(ctrl)
                qc.u(random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), ctrl)
    # print(qc.draw())
    # qc.save_statevector()
    # print(f"[DEBUG] generate random: #qubits: {qc.num_qubits}, depth: {qc.depth()}")
    return qc

def dump_qasm(qc, filename):
    # save the quantum circuit to a QASM file
    qasm_str = qasm2.dumps(qc)
    # 将qasm_str里面的表达式计算出来
    with open(filename, "w") as f:
        f.write(qasm_str)
    return

def dump_sv(sv, filename):
    with open(filename, "w") as f:
        for e in np.asarray(sv):
            f.write(str(e) + '\n')
    return

def load_qasm(filename):
    qc = QuantumCircuit.from_qasm_file(filename)
    return qc

# 获取当前节点的IP地址
def get_ip():
    try:
        # 连接到公共DNS服务器以获取本地IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # 备用方法：获取主机名对应的IP
        return socket.gethostbyname(socket.gethostname())

def myIQP(num_qubits):
    print(f"[IQP] #Qubits: [{num_qubits}]")
    A = np.random.randint(0, 10, size=(num_qubits, num_qubits))
    symmetric_matrix = (A + A.T) // 2 # 生成对称矩阵
    # print(symmetric_matrix)
    qc = IQP(symmetric_matrix).decompose()
    return qc

def VQC_AA(numQubits):
    qc = QuantumCircuit(numQubits)
    for _ in range(1):
        # 对每个量子比特应用随机的 RX 门
        for i in range(numQubits):
            angle_rx = np.random.rand() * 2 * pi  # 随机生成 0 到 2π 的角度
            qc.rx(angle_rx, i)

        # 对每个量子比特应用随机的 RZ 门
        for i in range(numQubits):
            angle_rz = np.random.rand() * 2 * pi  # 随机生成 0 到 2π 的角度
            qc.rz(angle_rz, i)

        # 双比特的 CRZ 门
        for i in range(numQubits):
            for j in range(numQubits):
                if i == j:
                    continue
                # qc.barrier()  # 加入 barrier 隔离量子门
                angle_crz = np.random.rand() * 2 * pi  # 随机生成 0 到 2π 的角度
                # qc.crz(angle_crz, i, j)  # 应用 CRZ 门
                qc.cx(i, j)
    return qc
