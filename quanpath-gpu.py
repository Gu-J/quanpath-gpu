import numpy as np
import time
import multiprocessing as mp
import psutil, os, gc
import argparse
import math
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from circuits import *
from trans import trans
from multiprocessing import shared_memory
import threading

class HybridQuantumSimulator:
    def __init__(self, high_indices, low_indices):
        self.high_indices = high_indices  # 存储高阶量子位索引
        self.low_indices = low_indices  # 存储低阶量子位索引
    

    def split_circuit(self, qc):
        """
        
        参数：
            qc: QuantumCircuit, 原始量子线路
        返回：
        
        """
        high_qubit = len(self.high_indices)
        low_qubit = len(self.low_indices)
        # 创建高位子线路，低阶
        high_qc = QuantumCircuit(high_qubit)
        low_qc_list = {i: QuantumCircuit(low_qubit) for i in range(2**high_qubit)}

        # 初始化低阶线路的状态向量
        # initsv = np.zeros(1 << low_qc_list[0].num_qubits, dtype=complex)
        # initsv[0] = 1 / math.sqrt(2**high_qubit)
        # for i in range(len(low_qc_list)):
        #     low_qc_list[i].set_statevector(initsv)
            

        # 常见控制门列表
        control_gates = ['cx','cy', 'cz', 'ccx', 'cswap','ch']
        # 遍历原始线路的操作
        for instruction in qc.data:
            gate = instruction.operation
            if gate.name == 'initialize' or gate.name == 'barrier':
                continue
            qubits = [qc.find_bit(q).index for q in instruction.qubits]
            # 检查是否所有量子比特都在高位子集
            if all(q in self.high_indices for q in qubits):
                new_qubits = [self.high_indices.index(q) for q in qubits]
                high_qc.append(gate, new_qubits)
            # 检查是否所有量子比特都在low
            elif all(q in self.low_indices for q in qubits):
                new_qubits = [self.low_indices.index(q) for q in qubits]
                for i in range(len(low_qc_list)):
                    low_qc_list[i].append(gate, new_qubits)
            # 检查是否是控制门，控制位在高位，目标位不在高位
            else:
                if gate.name in control_gates:
                    control_qubits = qubits[:-1]
                    target_qubit = qubits[-1]
                    # 控制在高位，目标在地位
                    if all(q in self.high_indices for q in control_qubits) and target_qubit not in self.high_indices:
                        new_gate = Gate(name=gate.name[1:], num_qubits=1, params=[])
                        new_target_qubit = [self.low_indices.index(target_qubit)]
                        for i in range(len(low_qc_list)):
                            # if all((i >> self.high_indices.index(q)) & 1 for q in control_qubits):
                            if all((i << low_qubit) >> q & 1 == 1  for q in control_qubits):
                                low_qc_list[i].append(new_gate, new_target_qubit)
                    # 控制在地位，目标在高位
                    else:
                        pass
                else:
                    raise Exception("非可分离线路！")


        return high_qc, low_qc_list
        
    def simulate_high_order(self, high_qc):
        """
        使用张量积和矩阵乘法计算高阶门的大矩阵
        """
        high_order_matrix = Operator(high_qc).data
        # print(high_qc)
        # print(high_order_matrix)
        return high_order_matrix

    def simulate_low_order(self, high_order_matrix, low_qc_list,fromzero,res):
        """
        使用 cuQuantum 的 statevector 模拟低阶门，并合并状态向量。
        
        Args:
            high_order_matrix
            low_qc_list (list): 子线路
        
        Returns:
            low_order_statevector:  2^n 维状态向量
        """
        high_qubit = len(self.high_indices)
        low_qubit = len(self.low_indices)
        self.nqubit=high_qubit+low_qubit
        chunk_len = 2**low_qubit
        res_len=chunk_len * 2**high_qubit 

        process = psutil.Process(os.getpid())
        print("\n--- 低阶部分 ---")
        print(f"内存占用: {process.memory_info().rss / 1024**3:.4f} GB")


        # --- 模拟第一块 ---
        print("\n--- 模拟第一块 ---")
        simulator = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)

        low_qc_list[0].save_statevector()
        start0 = time.time()
        job = simulator.run(low_qc_list[0])
        result = job.result()
        # 获取第一个状态向量
        current_statevector_view = result.get_statevector().data 
        end = time.time()
        print(f"内存占用 获取第一个状态向量 后): {process.memory_info().rss / 1024**3:.4f} GB")
        print(f'第一块SVSim耗时: {end - start0:.4f} 秒')

        # --- 循环处理后续的量子电路模拟和合并 ---
        if not fromzero:
            for k in range(len(low_qc_list) - 1):
                block_idx = k + 1 # 当前处理的 low_qc_list 索引 (1, 2, 3... len(low_qc_list) - 1)
                
                print(f"\n--- 处理第 {block_idx + 1} 块 ---") 

                start = time.time()
                processes = []
                for i in range(2**high_qubit): # 创建 4 个进程
                    p = threading.Thread(   target=self.merge,
                                            args=(i, high_order_matrix[i][len(low_qc_list) - 1],
                                                    current_statevector_view[0:chunk_len], chunk_len, res[i*chunk_len:(i+1)*chunk_len]))
                    p.start()
                    processes.append(p)

                # 同时运行当前块的量子模拟
                low_qc_list[block_idx].save_statevector()
                job = simulator.run(low_qc_list[block_idx])
                result = job.result()

                # 等待所有合并进程完成
                for p in processes:
                    p.join()
                
                current_statevector_view = result.get_statevector().data # 获取本次模拟得到的状态向量数据

                end = time.time()
                print(f"内存占用 获取状态向量 后): {process.memory_info().rss / 1024**3:.4f} GB")
                print(f'第 {block_idx + 1} 块SVSim和merge耗时: {end - start:.4f} 秒')


        # 最后一次合并操作,对应最后一块
        print("\n--- 合并最后一块 ---")
        start = time.time()
        processes = []
        for i in range(2**high_qubit):
            p = threading.Thread(   target=self.merge,
                                    args=(i, high_order_matrix[i][len(low_qc_list) - 1],
                                            current_statevector_view[0:chunk_len], chunk_len, res[i*chunk_len:(i+1)*chunk_len]))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        end = time.time()

        print(f'最后一次合并耗时: {end - start:.4f} 秒')
        print(f"内存占用: {process.memory_info().rss / 1024**3:.4f} GB")

        print(f'总计算耗时: {end - start0:.4f} 秒')

        # return res
        return end - start0
    

    def merge(self, rank, scalar, statevector, chunk_len, res):   # TODO
        start=time.time()
        num_t=16
        processes = []
        for i in range(num_t):
            slice_len=chunk_len//num_t
            p = threading.Thread(target=self.merge_mt,
                            args=(rank,i, scalar,statevector[i*slice_len:(i+1)*slice_len], res[i*slice_len:(i+1)*slice_len]))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("[子线程]",rank,"CPU 任务完成,",time.time()-start,'s')
    
    def merge_mt(self,rank, i,scalar, statevector, res):
        # start=time.time()
        if self.nqubit<=31:
            res[:] += statevector * scalar
        else:
            np.multiply(statevector, scalar, out=res)  # FIXME: 模拟内存充足情况
            np.add(res, statevector, out=res) 

        # print("[子线程]",rank,i,"CPU 任务完成,",time.time()-start,'s')

        
    def simulate(self, circuit,fromzero,res):
        """
        主模拟函数
        """
        print(f'量子线路 {circuit.num_qubits} 比特')

        # 1. 分割电路
        start = time.time()
        high_qc, low_qc_list = self.split_circuit(circuit)
        end = time.time()
        print(f'分割电路耗时: {end - start:.4f} 秒')
        
        # 2. 模拟高阶部分
        start = time.time()
        high_order_matrix = self.simulate_high_order(high_qc)
        end = time.time()
        print(f'高阶模拟耗时: {end - start:.4f} 秒')

        # 3. SVSim and merge
        start = time.time()
        result_time = self.simulate_low_order(high_order_matrix, low_qc_list,fromzero,res)

        # simulator = AerSimulator(method='statevector', device='CPU')
        # circuit.save_statevector()
        # job = simulator.run(circuit)
        # result = job.result()
        # sv_data = result.get_statevector().data

        end = time.time()

        print(f'低阶模拟总共耗时: {end - start:.4f} 秒')
        result_time=end - start
        return result_time

def create_separated_circuit(n_low, n_high,depth):
    """
    Generate a quantum circuit with separated low and high order qubits.
    
    Args:
        n_low (int): Number of low-order qubits
        n_high (int): Number of high-order qubits
    
    Returns:
        QuantumCircuit: Circuit with separated low/high qubit operations
    """
    n_total = n_low + n_high
    circuit = QuantumCircuit(n_total)
    for j in range(depth):
        # Apply gates on low-order qubits (q[0] to q[n_low-1])
        for i in range(n_low):
            circuit.h(i)  # Apply Hadamard to low-order qubits
            if i < n_low - 1:
                circuit.cx(i, i + 1)  # CX within low-order qubits
        
        # Apply gates on high-order qubits (q[n_low] to q[n_total-1])
        for i in range(n_low, n_total):
            circuit.x(i)  # Apply X gate to high-order qubits
            if i < n_total - 1:
                circuit.cz(i, i + 1)  # CZ within high-order qubits
        
        # Apply allowed cross-boundary gates (high-order as control, low-order as target)
        for i in range(n_low, n_total):
            circuit.cx(i, 0)  # High-order qubit controls low-order qubit q[0]
    return circuit

def baseline(args):
    simulator = AerSimulator(method='statevector', device=args.baseline, cuStateVec_enable=True)
    qc=generate_circuit(args,False)
    # qc=create_separated_circuit(args.low,args.high,args.depth)
    qc.save_statevector()
    start=time.time()
    job = simulator.run(qc)
    result = job.result()
    sv_data = result.get_statevector().data
    return time.time()-start


def meminit(args):
    process = psutil.Process(os.getpid())
    # --- Memory Init (共享内存初始化) ---
    start=time.time()
    high_qubit = args.high
    low_qubit = args.low
    chunk_len = 2**low_qubit
    # 用于存储当前状态向量的共享内存视图
    # current_statevector_view = np.ones(chunk_len , dtype=np.complex128) -1    # 立即分配
    # print(f"内存占用 (初始化共享 SV 后): {process.memory_info().rss / 1024**3:.4f} GB")

    # 用于存储最终结果的共享内存视图
    res = np.ones(chunk_len * 2**high_qubit, dtype=np.complex128) - 1
    print(f"内存占用 (初始化共享结果后): {process.memory_info().rss / 1024**3:.4f} GB")
    end=time.time()
    print(f'Mem init耗时: {end - start:.4f} 秒')
    return res
    

# 使用示例
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", type=int, default=2)
    parser.add_argument("--low", type=int, default=28)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--circuit", type=str, default='Random')
    parser.add_argument("--baseline", type=str, default='')
    parser.add_argument('--fromzero', action='store_true')
    args = parser.parse_args()
    args.nqubits=args.high+args.low
    fromzero = 1 if args.fromzero else 0

    if args.baseline != '':
        return baseline(args)

    # 创建量子线路
    highQubits = args.high
    lowQubits = args.low
    numQubits = highQubits + lowQubits
    # numBlocks = highQubits ** 2
    
    qc = trans(args)
    
    # qc = create_separated_circuit(lowQubits, highQubits,args.depth)
    
    # qc=[generate_circuit(args,False)]
    # qc=[qc]
    # with open("circuit.txt", "w") as f:
    #     f.write(qc[0].draw(output='text', fold=-1).single_string())


    res=meminit(args)

    i=0
    result_time=0
    for e in qc:
        i+=1
        print('!'*100,'sub circuit',i)
        simulator = HybridQuantumSimulator(list(range(lowQubits, numQubits)), list(range(lowQubits)))
        result_time += simulator.simulate(e,i==1 and fromzero,res)
        del simulator
        gc.collect()
        print('@'*100,'time',result_time)

    return result_time
    

if __name__ == "__main__":
    import sys
    original_stdout = sys.stdout
    with open('output.txt', "w") as f:
        sys.stdout = f
        result_time=main()
        sys.stdout = original_stdout
        print(result_time)