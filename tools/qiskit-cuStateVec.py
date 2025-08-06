# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers.jobstatus import JobStatus
import argparse
import math
import time
import numpy as np
import multiprocessing as mp
import psutil, os, gc

def create_ghz_circuit(n_qubits):
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for i in range(10):
        for qubit in range(n_qubits-1):
            circuit.cx(qubit, qubit + 1)
            circuit.h(qubit)
        # circuit.barrier()
    return circuit

def my_circuit(numQubits):
    numThreads = 32
    numHighQubits = int(math.log2(numThreads))
    numLowQubits = int(numQubits - numHighQubits)
    qc = QuantumCircuit(numQubits)
    for layer in range(1000):
        if layer % 5 == 0:
            for i in range(numQubits - 1, numLowQubits, -2):
                qc.h(i)
            for i in range(1, numLowQubits, 2):
                qc.cy(i, i - 1)
        elif layer % 5 == 1:
            for i in range(numQubits):
                if i % 3 == 0:
                    qc.rx((i + 1) / numQubits, i)
                if i % 3 == 1:
                    qc.ry((i + 1) / numQubits, i)
                if i % 3 == 2:
                    qc.rz((i + 1) / numQubits, i)
        elif layer % 5 == 2:
            for i in range(numQubits):
                if i % 3 == 0:
                    qc.x(i)
                if i % 3 == 1:
                    qc.y(i)
                if i % 3 == 2:
                    qc.z(i)
        elif layer % 5 == 3:
            for i in range(numQubits):
                if i >= numLowQubits:  # 高阶部分
                    qc.rx((layer + 1) / 200.0, i)
                else:  # 低阶部分
                    qc.ry((layer + 1) / 200.0, i)
        else:
            for i in range(numQubits - 1, numLowQubits, -2):
                qc.h(i);
            for i in range(1, numLowQubits, 2):
                qc.cz(i, i - 1)
        qc.barrier()

    return qc

def warmup(simulator):
    start = time.time()
    warmup = QuantumCircuit(2)
    warmup.h(0)
    warmup.cx(0, 1)
    warmup.save_statevector()
    simulator.run(transpile(warmup, simulator)).result()  # 这一步只是预热
    print('warm up,',time.time()-start,'s')


# def cpu_task(statevector0, chunk_len, result_array):
#     print("[子进程] CPU 任务开始")
#     start=time.time()
#     for i in range(4):
#         start_idx = chunk_len * i
#         end_idx = chunk_len * (i + 1)
#         result_array[start_idx:end_idx] += statevector0 * (i + 1.0)
#     print("[子进程] CPU 任务完成,",time.time()-start,'s')

def cpu_task_mp(rank,scalar,src, chunk_len, res):   # TODO
    print("[子进程]",rank,"CPU 任务开始")
    start=time.time()
    start_idx = chunk_len * rank
    end_idx = chunk_len * (rank + 1)
    res[start_idx:end_idx] += src * scalar
    print("[子进程]",rank,"CPU 任务完成,",time.time()-start,'s')
    
def copy_and_del(src,dest,ori): # TODO
    start=time.time()
    np.copyto(dest, src)
    del src
    del ori
    gc.collect()
    print('copy and del done,',time.time()-start,'s')



def run(n_qubits, precision, use_cusvaer):
    simulator = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=use_cusvaer)
    # simulator = AerSimulator(method='statevector', device='CPU')
    simulator.set_option('precision', precision)


    warmup(simulator)

    process = psutil.Process(os.getpid())
    print(f"内存占用: {process.memory_info().rss / 1024**3:.2f} GB")

    circuit = create_ghz_circuit(n_qubits)
    # circuit = my_circuit(n_qubits)
    # circuit.measure_all()
    circuit = transpile(circuit, simulator)
    with open("circuit.txt", "w") as f:
        f.write(circuit.draw(output="text").single_string())

    circuit.save_statevector()  # 保存状态向量

    start = time.time()
    job0 = simulator.run(circuit)
    result0 = job0.result()
    end = time.time()
    statevector0 = result0.get_statevector().data
    print((statevector0[:8]))
    print(f'模拟完成：{n_qubits} 比特, precision = {precision}, cusvaer = {use_cusvaer}')
    print(f'耗时: {end - start:.4f} 秒')
    print(f'backend: {result0.backend_name}')
    # print(circuit.draw())
    # print("最终状态向量为：")
    # for i, amp in enumerate(statevector):
    #     print(f'|{format(i, f"0{n_qubits}b")}>: {amp}')
    print(f"内存占用: {process.memory_info().rss / 1024**3:.2f} GB")
########### init
    chunk_len=statevector0.size
    total_len = statevector0.size * 4
    shared_statevector = mp.Array('d', chunk_len * 2)  # 每个 complex64 有两个 float
    statevector = np.frombuffer(shared_statevector.get_obj(), dtype=np.complex128).reshape((chunk_len,))
    print(f"内存占用: {process.memory_info().rss / 1024**3:.2f} GB")
    copy_and_del(statevector0,statevector,result0) # 0.8s
    print(f"内存占用: {process.memory_info().rss / 1024**3:.2f} GB")
    shared_res = mp.Array('d', total_len * 2)  # 每个 complex64 有两个 float
    res = np.frombuffer(shared_res.get_obj(), dtype=np.complex128).reshape((total_len,))
##########
    print(f"内存占用: {process.memory_info().rss / 1024**3:.2f} GB")

    start = time.time()

    processes = []
    for i in range(4):  # 创建 4 个进程
        p = mp.Process(target=cpu_task_mp, args=(i,2.0,statevector, chunk_len, res))
        processes.append(p)
        p.start()

    job1 = simulator.run(circuit)
    result1 = job1.result()

    for p in processes:
        p.join()  # 等待所有进程完成



    end = time.time()

    statevector1 = result1.get_statevector().data
    print((res[:8]))
    print((statevector1[:8]))
    print(f'模拟完成：{n_qubits} 比特, precision = {precision}, cusvaer = {use_cusvaer}')
    print(f'耗时: {end - start:.4f} 秒')
    print(f"内存占用: {process.memory_info().rss / 1024**3:.2f} GB")

    start = time.time()
    copy_and_del(statevector1,statevector,result1) # 0.8s
    print(f"内存占用: {process.memory_info().rss / 1024**3:.2f} GB")


    processes = []
    for i in range(4):  # 创建 4 个进程
        p = mp.Process(target=cpu_task_mp, args=(i,i+1.0,statevector, chunk_len, res))
        processes.append(p)
        p.start()

    job2 = simulator.run(circuit)
    result2 = job2.result()

    for p in processes:
        p.join()  # 等待所有进程完成



    end = time.time()

    statevector2 = result2.get_statevector().data
    print((res[:8]))
    print((statevector2[:8]))
    print(f'模拟完成：{n_qubits} 比特, precision = {precision}, cusvaer = {use_cusvaer}')
    print(f'耗时: {end - start:.4f} 秒')
    print(f"内存占用: {process.memory_info().rss / 1024**3:.2f} GB")



parser = argparse.ArgumentParser(description="Qiskit ghz.")
parser.add_argument('--nbits', type=int, default=28, help='the number of qubits')
parser.add_argument('--precision', type=str, default='double', choices=['single', 'double'], help='numerical precision')
parser.add_argument('--disable-cusvaer', default=False, action='store_true', help='disable cusvaer')

args = parser.parse_args()

run(args.nbits, args.precision, not args.disable_cusvaer)
# run(args.nbits, args.precision, not args.disable_cusvaer)
