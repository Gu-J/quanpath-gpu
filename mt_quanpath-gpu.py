import numpy as np
import time
import multiprocessing as mp
import psutil, os, gc
import argparse
import math
import threading
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from qiskit import transpile

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
        control_gates = ['cx','cy', 'cz', 'ccx', 'cswap']
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
                    print("非可分离线路！")
                    exit()

        return high_qc, low_qc_list
        
    def simulate_high_order(self, high_qc):
        """
        使用张量积和矩阵乘法计算高阶门的大矩阵
        """
        high_order_matrix = Operator(high_qc).data
        # print(high_qc)
        # print(high_order_matrix)
        return high_order_matrix

    def simulate_low_order(self, high_order_matrix, low_qc_list):
        """
        使用 cuQuantum 的 statevector 模拟低阶门，并合并状态向量。
        
        Args:
            high_order_matrix
            low_qc_list (list): 子线路
        
        Returns:
            low_order_statevector:  2^n 维状态向量
        """

        process = psutil.Process(os.getpid())
        print("\n--- 低阶部分 ---")
        print(f"内存占用: {process.memory_info().rss / 1024**3:.4f} GB")

        # --- Memory Init (共享内存初始化) ---
        start=time.time()
        high_qubit = len(self.high_indices)
        low_qubit = len(self.low_indices)
        chunk_len = 2**low_qubit
        # 用于存储当前状态向量的共享内存视图
        current_statevector_view = np.ones(chunk_len , dtype=np.complex128)     # 立即分配
        print(f"内存占用 (初始化共享 SV 后): {process.memory_info().rss / 1024**3:.4f} GB")

        # 用于存储最终结果的共享内存视图
        res = np.ones(chunk_len * 2**high_qubit, dtype=np.complex128) - 1
        print(f"内存占用 (初始化共享结果后): {process.memory_info().rss / 1024**3:.4f} GB")
        end=time.time()
        print(f'Mem init耗时: {end - start:.4f} 秒')

        # --- 模拟第一块 ---
        start0 = time.time()
        # print("\n--- 模拟第一块 ---")
        # simulator = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)

        # low_qc_list[0].save_statevector()
        # job = simulator.run(low_qc_list[0])
        # result = job.result()
        # # 获取第一个状态向量
        # current_sv_data = result.get_statevector().data 
        # print(f"内存占用 获取第一个状态向量 后): {process.memory_info().rss / 1024**3:.4f} GB")
        # end = time.time()
        # print(f'第一块SVSim耗时: {end - start0:.4f} 秒')

        # # 将第一个状态向量复制到共享内存视图中
        # start=time.time()
        # np.copyto(current_statevector_view, current_sv_data)
        # print('copy',time.time()-start)
        # start=time.time()
        # # 及时释放不再需要的临时对象
        # del current_sv_data
        # del result
        # del job
        # del simulator
        # gc.collect() 
        # print('del',time.time()-start)
        # print(f"内存占用 释放不再需要的临时对象 后): {process.memory_info().rss / 1024**3:.4f} GB")

        # --- 循环处理后续的量子电路模拟和合并 ---
        if(0):
            for k in range(len(low_qc_list) - 1):
                block_idx = k + 1 # 当前处理的 low_qc_list 索引 (1, 2, 3... len(low_qc_list) - 1)
                
                print(f"\n--- 处理第 {block_idx + 1} 块 ---") 

                start = time.time()
                processes = []
                for i in range(2**high_qubit): # 创建 4 个进程
                    p = threading.Thread(target=self.merge,
                                args=(i, high_order_matrix[i][k],
                                        current_statevector_view, chunk_len, res))
                    processes.append(p)
                    p.start()


                # 同时运行当前块的量子模拟
                simulator = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)
                low_qc_list[block_idx].save_statevector()
                job = simulator.run(low_qc_list[block_idx])
                result = job.result()
                next_sv_data = result.get_statevector().data # 获取本次模拟得到的状态向量数据

                # 等待所有合并进程完成
                for p in processes:
                    p.join()

                end = time.time()
                print(f"内存占用 获取状态向量 后): {process.memory_info().rss / 1024**3:.4f} GB")
                print(f'第 {block_idx + 1} 块SVSim和merge耗时: {end - start:.4f} 秒')

                # 将本次模拟得到的最新状态向量复制到共享内存中，供下一轮循环使用
                start = time.time()
                np.copyto(current_statevector_view, next_sv_data)
                del next_sv_data
                del result
                del job
                del simulator
                gc.collect()
                print('copy and del done,',time.time()-start,'s')
                print(f"内存占用 释放不再需要的临时对象 后): {process.memory_info().rss / 1024**3:.4f} GB")
                # print(res)

        # 最后一次合并操作,对应最后一块
        print("\n--- 合并最后一块 ---")
        start = time.time()
        processes = []
        for i in range(2**high_qubit):
            p = threading.Thread(target=self.merge,
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

        print("\n--- final merge ---")
        del current_statevector_view
        gc.collect()
        res1 = np.ones(chunk_len * 2**high_qubit, dtype=np.complex128) - 1
        print(f"内存占用 (初始化共享结果后): {process.memory_info().rss / 1024**3:.4f} GB")
        start = time.time()
        processes = []
        num_t=64
        slice_len=chunk_len//num_t
        for i in range(num_t):
            p = threading.Thread(target=self.final_merge,
                            args=(i,2**high_qubit, high_order_matrix[0:],chunk_len,slice_len, res[0:],res1[0:]))
            p.start()
            processes.append(p)


        for p in processes:
            p.join()
        end = time.time()

        print(f'final merge耗时: {end - start:.4f} 秒')
        print(f"内存占用: {process.memory_info().rss / 1024**3:.4f} GB")

        return res

    # def gpu_svsim(qc):
    #     simulator = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)
    #     qc.save_statevector()
    #     job = simulator.run(qc)
    #     result = job.result()
    #     next_sv_data = result.get_statevector().data
    #     np.copyto(current_statevector_view, next_sv_data)


    def final_merge(self, rank, num_chunks,high_order_matrix,  chunk_len, slice_len,res,res1):   # TODO
        start=time.time()
        for i in range(num_chunks):
            res_ind=i*chunk_len+rank*slice_len
            for j in range(num_chunks):
                offset=j*chunk_len+rank*slice_len
                res1[res_ind:res_ind+slice_len]+=res[offset:offset+slice_len]*high_order_matrix[i][j]
        # print("[子线程]",rank,"CPU 任务完成,",time.time()-start,'s')

    def merge(self, rank, scalar, statevector, chunk_len, res):   # TODO
        start=time.time()
        processes = []
        num_t=16
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
        res[:] += statevector * scalar
        # print("[子线程]",rank,i,"CPU 任务完成,",time.time()-start,'s')

        
    def simulate(self, circuit):
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
        low_order_statevector = self.simulate_low_order(high_order_matrix, low_qc_list)
        end = time.time()

        print(f'低阶模拟总共耗时: {end - start:.4f} 秒')

        # 4. 合并结果
        # start = time.time()
        # final_result = self.merge_results(high_order_matrix, low_order_statevector)
        # end = time.time()

        # print(f'耗时: {end - start:.4f} 秒')

        # 将 final_statevector 保存到文本文件
        # with open("final_statevector.txt", 'w') as f:
        #     for i, amplitude in enumerate(low_order_statevector):
        #         f.write(f"{amplitude.real:.6f} {amplitude.imag:.6f}\n")
                
        return 

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

def baseline(qc,dev):
    start=time.time()
    simulator = AerSimulator(method='statevector', device=dev, cuStateVec_enable=True)
    qc.save_statevector()
    # qc=transpile(qc, optimization_level=3)
    job = simulator.run(qc)
    result = job.result()
    sv_data = result.get_statevector().data
    print('baseline,',time.time()-start,'s')
    exit()


# 使用示例
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", type=int, default=2)
    parser.add_argument("--low", type=int, default=26)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--baseline", type=str, default='')
    args = parser.parse_args()

    # 创建量子线路
    highQubits = args.high
    lowQubits = args.low
    numQubits = highQubits + lowQubits
    # numBlocks = highQubits ** 2
    
    qc = create_separated_circuit(lowQubits, highQubits,args.depth)

    # with open("circuit.txt", "w") as f:
    #     f.write(qc.draw(output="text").single_string())

    if args.baseline != '':
        baseline(qc,args.baseline)

    # 创建模拟器实例
    simulator = HybridQuantumSimulator(list(range(lowQubits, numQubits)), list(range(lowQubits)))

    
    # 运行模拟
    result = simulator.simulate(qc)
    

if __name__ == "__main__":
    main()
