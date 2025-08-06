import networkx as nx
import numpy as np
import copy
import time
import sys
from collections import defaultdict
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_aer import AerSimulator
import argparse
from circuits import *

def print_rank0(str,file=''):
    # print(str)
    return

class QuanTrans:
    def __init__(self):
        return
    
    def transform(self, circ, numhq):
        self.circ = circ
        self.numhq = numhq
        self.numlq = self.circ.num_qubits - numhq
        self.dag  = circuit_to_dag(circ)
        self.remove_single_qubit_gates()

        # partition table
        print_rank0(f"[DEBUG] build_partition_table", file=sys.stderr)
        self.build_partition_table()

        # slicing table and slicing results
        print_rank0(f"[DEBUG] build_slicing_table", file=sys.stderr)
        self.build_slicing_table()
        self.SubC = []
        self.get_sliced_subc(0, len(self.P)-1) # -> self.SubC
        print_rank0(f"[INFO] #SubCs: {len(self.SubC)}")

        return self.reorder_qubits()

    def remove_single_qubit_gates(self):
        """
        Remove single qubit gates to construct self.dag_multiq
        """
        # 根据原始线路的dag，逐层保留双量子比特门，并记录双量子比特门的层号
        start_time = time.time()
        self.dag_multiq = []
        self.map_to_init_layer = {}
        self.map_to_dag_multi_layer = {}
        # 遍历self.dag的每一层，如果是双量子比特门
        # 则添加到self.dag_multiq
        layers = list(self.dag.layers())
        for lev, layer in enumerate(layers):
            curr_layer = []
            for node in layer["graph"].op_nodes():
                # print_rank0(f"{lev} {node.op.name} {node.qargs}")
                if len(node.qargs) > 1:
                    # if node.op.name == "barrier":
                    #     continue
                    # if len(node.qargs) != 2:
                    #     print_rank0(node.op.name)
                    assert len(node.qargs) == 2, f"[ERROR] {node.op.name} has more than 2 input qubits"
                    curr_layer.append(node)
                    # dag_debug.apply_operation_back(node.op, node.qargs, node.cargs)
            if len(curr_layer) > 0:
                self.dag_multiq.append(curr_layer)
                # 记录双量子比特门在原始线路的层号
                self.map_to_init_layer[len(self.dag_multiq)-1] = lev
                self.map_to_dag_multi_layer[lev] = len(self.dag_multiq)-1
        # print_rank0(self.map_to_init_layer)

        # subc = dag_to_circuit(dag_debug)
        # print_rank0("[remove_single_qubit_gates]")
        # print_rank0(subc)
        # for lev, nodes in enumerate(self.dag_multiq):
        #     print_rank0(f"lev {lev}")
        #     for node in nodes:
        #         print_rank0(node.op.name, node.qargs)
        end_time = time.time()
        print_rank0(f"[DEBUG] remove_single_qubit_gates: {end_time - start_time} seconds", file=sys.stderr)
        return

    def build_partition_table(self):
        """
        An efficient way of building the partition table
        """
        start_time = time.time()
        num_depths = len(self.dag_multiq)
        # print_rank0(f"[DEBUG] num_depths: {num_depths}", file=sys.stderr)
        self.P = [[set() for _ in range(num_depths)] for _ in range(num_depths)]
        cnt = 0
        # build the qubit interaction nxGraph for the entire circuit
        # print_rank0(f"[DEBUG] build_qubit_interaction_graph", file=sys.stderr)
        qig = self.build_qubit_interaction_graph((0, num_depths-1))
        is_changed = True

        for i in range(num_depths):
            # if i % 100 == 0:
            #     print_rank0(f"[DEBUG] build_partition_table depth [{i}]")
            # ===== P[i][numDepths-1] =====
            # rebuild qig
            if i != 0: # remove the (i-1)-th level of the remaining qig
                is_changed = self.remove_qig_edge(qig, i-1)

            if len(self.P[i][num_depths-1]) > 0: # inherit from the upper grid
                assert i != 0, f"[ERROR] P[{i}][{num_depths-1}] should be empty."
                success = True # leftward propagation
                if i + 1 < num_depths: # downward propagation
                    self.P[i+1][num_depths-1] = self.P[i][num_depths-1]
            else:
                success = False
                if is_changed:
                    self.P[i][num_depths-1] = self.get_qig_partitions(qig)
                    cnt += 1
                    if len(self.P[i][num_depths-1]) > 0:
                        success = True # leftward propagation
                        if i + 1 < num_depths:
                            self.P[i+1][num_depths-1] = self.P[i][num_depths-1] # downward propagation

            # ===== P[i][numDepths-2 ~ i] =====
            qig_tmp = qig.copy()
            for j in range(num_depths - 2, i - 1, -1):
                is_changed = self.remove_qig_edge(qig_tmp, j+1)
                # print_rank0(f"depth [{i}][{j}]", file=sys.stderr)
                if len(self.P[i][j]) > 0: # inherit from the upper grid
                    success = True # leftward propagation
                    if i + 1 <= j: # i + 1 < numDepths
                        self.P[i+1][j] = self.P[i][j] # downward propagation
                elif success: # inherit from the right grid
                    self.P[i][j] = self.P[i][j+1]
                else:
                    # print_rank0(f"is_changed: {is_changed}", file=sys.stderr)
                    if is_changed:
                        self.P[i][j] = self.get_qig_partitions(qig_tmp)
                        cnt += 1
                        if len(self.P[i][j]) > 0:
                            success = True # leftward propagation
                            if i + 1 <= j:
                                self.P[i+1][j] = self.P[i][j]
        end_time = time.time()
        print_rank0(f"[build_partition_table] Partition calculation times: {cnt}.")
        print_rank0(f"[build_partition_table] Time: {end_time - start_time} seconds")
        return

    def build_qubit_interaction_graph(self, level_range):
        G = nx.Graph()
        for qubit in range(self.circ.num_qubits):
            G.add_node(qubit)
        for lev in range(level_range[0], level_range[1]+1):
            for node in self.dag_multiq[lev]:
                qubits = [qubit._index for qubit in node.qargs]
                if qubits[0] == None:
                    qubits = [self.circ.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
                if G.has_edge(qubits[0], qubits[1]):
                    G[qubits[0]][qubits[1]]['weight'] += 1
                else:
                    G.add_edge(qubits[0], qubits[1], weight=1)
        return G

    def remove_qig_edge(self, qig, lev):
        """
        从qig中移除self.dag_multiq第lev列的量子门
        """
        is_changed = False # whether an edge is removed from qig
        for node in self.dag_multiq[lev]:
            qubits = [qubit._index for qubit in node.qargs]
            if qig.has_edge(qubits[0], qubits[1]):
                qig[qubits[0]][qubits[1]]['weight'] -= 1
                if qig[qubits[0]][qubits[1]]['weight'] == 0:
                    qig.remove_edge(qubits[0], qubits[1])
                    is_changed = True
        return is_changed

    def get_qig_partitions(self, qig):
        # start_time = time.time()
        components = [list(comp) for comp in nx.connected_components(qig)]
        # legal_partitions = self.partitioner.partition(components, self.qpus[0])
        # legal_partitions = self.partitioner.partition(components, self.partition_method)
        # end_time = time.time()
        # print_rank0(f"[get_qig_partitions] Time: {end_time - start_time} seconds")
        # return legal_partitions
        return self.group_components_with_sum(components)

    def group_components_with_sum(self, components):
        num_ucs = len(components)
        lengths = [len(arr) for arr in components]
        # dp[numUCs+1][h+1], dp[numUCs][h] is what we want
        # dp[i][j]    := valid P0 with top i UCs and |P0|=j, 1 <= i <= numUCs，1 <= j <= h
        # dp[i][j][t] := valid P0 with top i UCs and |P0|=j and t-th partition
        dp = [set() for _ in range(self.numhq + 1)]
        dp[0].add(tuple())  # 初始状态：空子集的和为0

        for i in range(num_ucs):
            for j in range(self.numhq, lengths[i] - 1, -1):
                for prev_subset in dp[j - lengths[i]]:
                    new_subset = prev_subset + (i,)  # 记录元素索引
                    dp[j].add(new_subset)

        result = []
        for part in dp[self.numhq]:
            flattened = set()
            for uc_idx in part:
                flattened.update(components[uc_idx])
            result.append(flattened)
        return max(result, default=set(), key=lambda s: tuple(sorted(s)))

    # 
    # S, T table
    # 
    def build_slicing_table(self):
        start_time = time.time()
        num_depths = len(self.P)
        self.T = [[0]  * num_depths for _ in range(num_depths)]
        self.S = [[-1] * num_depths for _ in range(num_depths)]

        for i in range(num_depths):
            if len(self.P[i][i]) == 0:
                print_rank0(f"[ERROR] P[{i}][{i}] is empty.")
                exit(1)
        # print_rank0("[build_t_table] ", end="")
        for depth in range(2, num_depths + 1): # depth: 2, 3, ..., num_depths
            # print_rank0(depth, end="")
            for i in range(0, num_depths - depth + 1): # 左边界
                j = i + depth - 1 # 右边界
                if len(self.P[i][j]) == 0:
                    self.T[i][j] = float('inf')
                    # 利用四边形优化缩小枚举范围
                    lower_k = self.S[i][j-1] if self.S[i][j-1] != -1 else i
                    upper_k = self.S[i+1][j] if self.S[i+1][j] != -1 else j-1
                    for k in range(lower_k, upper_k + 1):
                    # for k in range(i, j):
                        comms = self.T[i][k] + self.T[k+1][j] + 1
                        if comms < self.T[i][j]:
                            self.T[i][j] = comms
                            self.S[i][j] = k
                    # check if S[i][j-1] <= S[i][j] <= S[i+1][j]
                    # print_rank0(i, self.S[i][j-1], self.S[i][j], self.S[i+1][j], j)
                    # if self.S[i][j-1] != -1:
                    #     assert(self.S[i][j-1] <= self.S[i][j])
                    # if self.S[i+1][j] != -1:
                    #     assert(self.S[i][j] <= self.S[i+1][j])
        # print_rank0()
        end_time = time.time()
        print_rank0(f"[build_slicing_table] Time: {end_time - start_time} seconds")
        return

    def get_sliced_subc(self, i, j):
        if i > j:
            self.SubC.append(((-1, -1), set(range(self.numlq, self.circ.num_qubits))))
            return
        if self.S[i][j] == -1:
            self.SubC.append(((i, j), self.P[i][j]))
            return
        self.get_sliced_subc(i, self.S[i][j])
        self.get_sliced_subc(self.S[i][j] + 1, j)
        return

    # 
    # add single qubit gates
    # 
    def reorder_qubits(self):
        new_circ = QuantumCircuit(self.circ.num_qubits)
        self.num_swaps = 0
        mapping = [i for i in range(self.circ.num_qubits)]
        prev_p0 = set(range(self.numlq, self.circ.num_qubits))
        layers = list(self.dag.layers())
        i = 0

        for idx, (ran, p0) in enumerate(self.SubC):
            if idx == len(self.SubC) - 1: # 最后一个子线路
                j = self.circ.depth() - 1
            else:
                j = self.map_to_init_layer[ran[1]]
            curr_p0 = p0
            # print_rank0(f"SubC[{idx}] {i}-{j}: {p0}")
            # 获取在原线路中的范围
            assert len(prev_p0) == len(curr_p0), f"[ERROR] len: prev_p0 != curr_p0"
            # print_rank0(f"[DEBUG] ori_subc: {i}-{j}")
            # print_rank0(f"[DEBUG] p0: {prev_p0} -> {curr_p0}")

            # 统计需要SWAP的量子比特（原始顺序）
            unique_prev = list(prev_p0 - curr_p0)
            unique_curr = list(curr_p0 - prev_p0)
            assert len(unique_prev) == len(unique_curr), f"[ERROR] unique_prev: {unique_prev}, unique_curr: {unique_curr}"
            swap_pair = [(unique_prev[i], unique_curr[i]) for i in range(len(unique_prev))]
            # print_rank0(f"[DEBUG] swap_pair: {swap_pair}")

            # add swap gates
            if idx > 0:
                new_circ.barrier()
            for (added_qubit, removed_qubit) in swap_pair:
                add_index = mapping.index(added_qubit)
                remove_index = mapping.index(removed_qubit)
                mapping[add_index], mapping[remove_index] = mapping[remove_index], mapping[add_index]
                if idx > 0:  # 如果是第一个子线路（i=0），不添加SWAP门
                    new_circ.swap(add_index, remove_index)
                    self.num_swaps += 1
            if idx > 0:
                new_circ.barrier()

            # reorder gates
            prev_p0 = set()
            high_order_gates = []
            for level in range(i, j+1):
                # print_rank0(f"[DEBUG] level: {level}")
                gate_layer = layers[level]["graph"].op_nodes()
                for node in gate_layer:
                    qubits = [qubit._index for qubit in node.qargs]
                    if qubits[0] == None:
                        qubits = [self.circ.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
                    new_qubits = [mapping.index(qubit) for qubit in qubits]
                    # print_rank0(f"[DEBUG] mapping: {mapping}")
                    # print_rank0(f"[DEBUG] {node.op.name} {qubits}->{new_qubits}")
                    # 如果new_qubits在高位，那么先加到high_order_gates
                    if new_qubits[0] >= self.numlq:
                        prev_p0.update(qubits)
                        high_order_gates.append(node)
                    else:
                        new_circ.append(node.op, new_qubits)

            # # 如果有高位的量子门，再做swap
            # # print_rank0(f"[DEBUG] high_order_qubits: {prev_p0}")
            # # print_rank0(f"[DEBUG] high_order_gates: {high_order_gates}")
            # # 选len(high_order_qubits)个量子比特作为高位量子比特
            # curr_p0 = set(mapping[i] for i in range(len(prev_p0)))

            # # print_rank0(f"[DEBUG] p0: {prev_p0} -> {curr_p0}")

            # # 统计需要SWAP的量子比特（原始顺序）
            # unique_prev = list(prev_p0 - curr_p0)
            # unique_curr = list(curr_p0 - prev_p0)
            # assert len(unique_prev) == len(unique_curr), f"[ERROR] unique_prev: {unique_prev}, unique_curr: {unique_curr}"
            # swap_pair = [(unique_prev[i], unique_curr[i]) for i in range(len(unique_prev))]
            # # print_rank0(f"[DEBUG] swap_pair: {swap_pair}")

            # # add swap gates
            # new_circ.barrier()
            # for (added_qubit, removed_qubit) in swap_pair:
            #     add_index = mapping.index(added_qubit)
            #     remove_index = mapping.index(removed_qubit)
            #     mapping[add_index], mapping[remove_index] = mapping[remove_index], mapping[add_index]
            #     new_circ.swap(add_index, remove_index)
            #     self.num_swaps += 1
            # new_circ.barrier()

            # # 添加高位量子门
            # for node in high_order_gates:
            #     qubits = [qubit._index for qubit in node.qargs]
            #     if qubits[0] == None:
            #         qubits = [self.circ.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
            #     new_qubits = [mapping.index(qubit) for qubit in qubits]
            #     # print_rank0(f"[DEBUG] mapping: {mapping}")
            #     # print_rank0(f"[DEBUG] {node.op.name} {qubits}->{new_qubits}")
            #     new_circ.append(node.op, new_qubits)

            # prev_p0更新成现在mapping高位的self.numhq个
            prev_p0 = set(mapping[i] for i in range(self.numlq, self.circ.num_qubits))
            # print_rank0(f"[DEBUG] prev_p0 <- {prev_p0}")

            i = j + 1

        # 恢复原始的量子比特顺序
        # tqid = 0
        # new_circ.barrier()
        # for sqid_pos in range(len(mapping)):
        #     tqid_pos = mapping.index(tqid)
        #     if sqid_pos != tqid_pos:
        #         new_circ.swap(sqid_pos, tqid_pos)
        #         # 更新 mapping
        #         mapping[sqid_pos], mapping[tqid_pos] = mapping[tqid_pos], mapping[sqid_pos]
        #     tqid += 1
        # new_circ.barrier()
        print_rank0(f"[INFO] QuanTrans: #rswap: {self.num_swaps}")
        return new_circ



def trans(args):
    args.nqubits=args.high+args.low
    nhqubits = args.high

    qc = generate_circuit(args, False)
    # qc = load_qasm(args.qasm)
    # qc = transpile(qc, dis_simulator, optimization_level=0)
    with open("circuit.txt", "w") as f:
        f.write(qc.draw(output='text', fold=-1).single_string())

    trans = QuanTrans()
    subcircuits = trans.transform(qc, nhqubits)
    with open("circuit.txt", "a") as f:
        f.write('\n'+'*'*100+'\n')
        f.write(subcircuits.draw(output='text', fold=-1).single_string())
    return subcircuits



if __name__ == "__main__":
    # get command line arguments
    # args = get_options()
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", type=int, default=2)
    parser.add_argument("--low", type=int, default=26)
    parser.add_argument("--circuit", type=str, default='qft')
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--baseline", type=str, default='')
    parser.add_argument('--fromzero', action='store_true')
    args = parser.parse_args() 
    trans(args)
