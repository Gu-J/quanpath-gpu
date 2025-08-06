#include <cuda_runtime.h>
#include <custatevec.h>
#include <cuComplex.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include "json.hpp"  // nlohmann/json库，需提前安装

using json = nlohmann::json;

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = call;                        \
    if (e != cudaSuccess) {                      \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at line " << __LINE__ << std::endl; \
        exit(1);                                \
    }                                           \
} while(0)

#define CHECK_CUSTATEVEC(call) do {              \
    custatevecStatus_t status = call;             \
    if (status != CUSTATEVEC_STATUS_SUCCESS) {   \
        std::cerr << "cuStateVec error: " << custatevecGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(1);                                 \
    }                                            \
} while(0)

const int NUM_QUBITS = 30;
const size_t STATE_SIZE = size_t(1) << NUM_QUBITS;  // 2^30

// 全局门矩阵变量
cuDoubleComplex* d_u_matrix = nullptr;
cuDoubleComplex* d_h_matrix = nullptr;
cuDoubleComplex* d_cx_matrix = nullptr;

// 单量子比特 U 门矩阵生成
void build_u_matrix(double theta, double phi, double lambda, cuDoubleComplex* mat) {
    double cos_t = cos(theta / 2.0);
    double sin_t = sin(theta / 2.0);
    mat[0] = make_cuDoubleComplex(cos_t, 0.0);
    mat[1] = make_cuDoubleComplex(-sin_t * cos(lambda), -sin_t * sin(lambda));
    mat[2] = make_cuDoubleComplex(sin_t * cos(phi), sin_t * sin(phi));
    double real = cos_t * cos(phi + lambda);
    double imag = cos_t * sin(phi + lambda);
    mat[3] = make_cuDoubleComplex(real, imag);
}

// 单量子比特 Hadamard 门矩阵
void build_h_matrix(cuDoubleComplex* mat) {
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    mat[0] = make_cuDoubleComplex(inv_sqrt2, 0.0);
    mat[1] = make_cuDoubleComplex(inv_sqrt2, 0.0);
    mat[2] = make_cuDoubleComplex(inv_sqrt2, 0.0);
    mat[3] = make_cuDoubleComplex(-inv_sqrt2, 0.0);
}

// 设备内存申请并拷贝辅助函数
cuDoubleComplex* deviceMallocAndCopy(const cuDoubleComplex* h_data, size_t length) {
    cuDoubleComplex* d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, length * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMemcpy(d_ptr, h_data, length * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    return d_ptr;
}

// 释放设备内存
void deviceFree(cuDoubleComplex* d_ptr) {
    if (d_ptr) {
        CHECK_CUDA(cudaFree(d_ptr));
    }
}

// 初始化所有门矩阵
void initialize_gate_matrices() {
    // 初始化U门矩阵（使用默认参数作为模板）
    cuDoubleComplex h_u[4];
    build_u_matrix(0.0, 0.0, 0.0, h_u);
    d_u_matrix = deviceMallocAndCopy(h_u, 4);

    // 初始化H门矩阵
    cuDoubleComplex h_h[4];
    build_h_matrix(h_h);
    d_h_matrix = deviceMallocAndCopy(h_h, 4);

    // 初始化CX门矩阵
    cuDoubleComplex h_cx[16] = {
        {1,0},{0,0},{0,0},{0,0},
        {0,0},{1,0},{0,0},{0,0},
        {0,0},{0,0},{0,0},{1,0},
        {0,0},{0,0},{1,0},{0,0}
    };
    d_cx_matrix = deviceMallocAndCopy(h_cx, 16);
}

// 应用单量子比特 U 门 (target qubit)
void apply_u(custatevecHandle_t handle, cuDoubleComplex* d_state, int target, const std::vector<double>& params) {
    // 为U门动态计算参数并更新设备上的矩阵
    cuDoubleComplex h_u[4];
    build_u_matrix(params[0], params[1], params[2], h_u);
    CHECK_CUDA(cudaMemcpy(d_u_matrix, h_u, 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    int32_t targetQubits[1] = { target };

    CHECK_CUSTATEVEC(custatevecApplyMatrix(
        handle,
        d_state,
        CUDA_C_64F,
        NUM_QUBITS,
        d_u_matrix,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,
        targetQubits,
        1,
        nullptr,
        nullptr,
        0,
        CUSTATEVEC_COMPUTE_64F,
        nullptr,
        0));
}

// 应用 Hadamard 门 (target qubit)
void apply_h(custatevecHandle_t handle, cuDoubleComplex* d_state, int target) {
    int32_t targetQubits[1] = { target };

    CHECK_CUSTATEVEC(custatevecApplyMatrix(
        handle,
        d_state,
        CUDA_C_64F,
        NUM_QUBITS,
        d_h_matrix,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,
        targetQubits,
        1,
        nullptr,
        nullptr,
        0,
        CUSTATEVEC_COMPUTE_64F,
        nullptr,
        0));
}

// 应用 CNOT 门 (control qubit -> target qubit)
void apply_cx(custatevecHandle_t handle, cuDoubleComplex* d_state, int control, int target) {
    int32_t targetQubits[1] = { target };
    int32_t controlQubits[1] = { control };
    int32_t controlBitValues[1] = { 1 };

    CHECK_CUSTATEVEC(custatevecApplyMatrix(
        handle,
        d_state,
        CUDA_C_64F,
        NUM_QUBITS,
        d_cx_matrix,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,
        targetQubits,
        1,
        controlQubits,
        controlBitValues,
        1,
        CUSTATEVEC_COMPUTE_64F,
        nullptr,
        0));
}

// 从设备读取状态向量并输出前N个元素
void print_statevector(custatevecHandle_t handle, cuDoubleComplex* d_state, int num_qubits, int num_elements = 16) {
    size_t state_size = size_t(1) << num_qubits;
    num_elements = std::min(num_elements, (int)state_size);
    
    // 计算需要提取的额外索引 (1<<28)
    const int special_index = 1 << 28;
    bool print_special = (special_index < state_size);
    
    // 确定需要从设备读取的元素数量（包括特殊索引）
    std::vector<int> indices_to_read;
    for (int i = 0; i < num_elements; i++) {
        indices_to_read.push_back(i);
    }
    if (print_special) {
        indices_to_read.push_back(special_index);
    }
    
    // 分配内存并读取指定索引的元素
    cuDoubleComplex* h_state = new cuDoubleComplex[indices_to_read.size()];
    size_t bytes_to_read = indices_to_read.size() * sizeof(cuDoubleComplex);
    
    // 使用cudaMemcpyAsync结合偏移量读取指定索引（更高效的方式）
    // 注意：这里需要先将状态向量映射到主机内存，然后直接访问指定索引
    cuDoubleComplex* mapped_state = nullptr;
    cudaError_t err = cudaHostAlloc(&mapped_state, state_size * sizeof(cuDoubleComplex), cudaHostAllocMapped);
    if (err == cudaSuccess) {
        CHECK_CUDA(cudaMemcpy(mapped_state, d_state, state_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        
        // 填充h_state数组
        for (size_t i = 0; i < indices_to_read.size(); i++) {
            h_state[i] = mapped_state[indices_to_read[i]];
        }
        
        CHECK_CUDA(cudaFreeHost(mapped_state));
    } else {
        // 回退到传统方式（适用于旧版CUDA）
        CHECK_CUDA(cudaMemcpy(h_state, d_state, state_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }
    
    // 输出前num_elements个元素
    std::cout << "\n状态向量前" << num_elements << "个元素:" << std::endl;
    for (int i = 0; i < num_elements; i++) {
        std::cout << "[" << std::setw(2) << std::setfill('0') << i << "] = (" 
                  << std::fixed << std::setprecision(6) << h_state[i].x << ", " 
                  << std::fixed << std::setprecision(6) << h_state[i].y << "i)" << std::endl;
    }
    
    // 输出特殊索引(1<<28)的元素
    if (print_special) {
        size_t special_idx_in_array = num_elements;
        std::cout << "\n特殊索引 [" << special_index << "] = (" 
                  << std::fixed << std::setprecision(6) << h_state[special_idx_in_array].x << ", " 
                  << std::fixed << std::setprecision(6) << h_state[special_idx_in_array].y << "i)" << std::endl;
    }
    
    delete[] h_state;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: ./test_cuquantum_30qubit_json circuit.json [输出元素数量]" << std::endl;
        return 1;
    }

    int output_elements = 16;
    if (argc >= 3) {
        output_elements = std::stoi(argv[2]);
    }

    // 读取 JSON 文件
    std::ifstream ifs(argv[1]);
    if (!ifs.is_open()) {
        std::cerr << "无法打开文件: " << argv[1] << std::endl;
        return 1;
    }
    json circuit_json;
    ifs >> circuit_json;

    // 初始化 cuStateVec
    custatevecHandle_t handle;
    CHECK_CUSTATEVEC(custatevecCreate(&handle));

    // 初始化门矩阵
    initialize_gate_matrices();

    // 初始化量子态 |0>
    cuDoubleComplex* d_state = nullptr;
    CHECK_CUDA(cudaMalloc(&d_state, STATE_SIZE * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMemset(d_state, 0, STATE_SIZE * sizeof(cuDoubleComplex)));
    cuDoubleComplex init_ampl = make_cuDoubleComplex(1.0, 0.0);
    CHECK_CUDA(cudaMemcpy(d_state, &init_ampl, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // 门计数器
    int h_count = 0;
    int cx_count = 0;
    int u_count = 0;

    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    // 逐门应用并统计
    for (const auto& gate : circuit_json) {
        std::string name = gate["name"];
        std::vector<int> qubits = gate["qubits"].get<std::vector<int>>();

        if (name == "u") {
            u_count++;
            std::vector<double> params = gate["params"].get<std::vector<double>>();
            if (qubits.size() != 1 || params.size() != 3) {
                std::cerr << "错误: U门参数不正确" << std::endl;
                continue;
            }
            apply_u(handle, d_state, qubits[0], params);
        } else if (name == "cx") {
            cx_count++;
            if (qubits.size() != 2) {
                std::cerr << "错误: CX门参数不正确" << std::endl;
                continue;
            }
            apply_cx(handle, d_state, qubits[0], qubits[1]);
        } else if (name == "h") {
            h_count++;
            if (qubits.size() != 1) {
                std::cerr << "错误: H门参数不正确" << std::endl;
                continue;
            }
            apply_h(handle, d_state, qubits[0]);
        } else {
            std::cerr << "不支持的门: " << name << std::endl;
        }
    }

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end - start).count();
    
    // 输出门统计信息
    std::cout << "\n门统计:" << std::endl;
    std::cout << "  H门数量: " << h_count << std::endl;
    std::cout << "  CX门数量: " << cx_count << std::endl;
    std::cout << "  U门数量: " << u_count << std::endl;
    std::cout << "  总门数: " << (h_count + cx_count + u_count) << std::endl;
    
    std::cout << "\n电路执行时间: " << elapsed_seconds << " 秒" << std::endl;

    // 输出状态向量
    print_statevector(handle, d_state, NUM_QUBITS, output_elements);

    // 清理资源
    deviceFree(d_u_matrix);
    deviceFree(d_h_matrix);
    deviceFree(d_cx_matrix);
    CHECK_CUDA(cudaFree(d_state));
    CHECK_CUSTATEVEC(custatevecDestroy(handle));

    return 0;
}