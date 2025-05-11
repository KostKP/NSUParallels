#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>
#include <cstdio>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

using namespace std;
using namespace chrono;

#define OUT_FILE "result.dat"

#ifndef NX
#define NX 30
#endif
#ifndef NY
#define NY 20
#endif

#define TAU -0.01
#define EPS 0.01
#define MAX_ITER 1000
#define SIZE ((size_t)(NX) * (size_t)(NY))
#define BLOCK_SIZE 256

__device__ double get_a(int row, int col) {
    if (row == col) return -4;
    if (row + 1 == col) return 1;
    if (row - 1 == col) return 1;
    if (row + NX == col) return 1;
    if (row - NX == col) return 1;
    return 0;
}

__device__ double get_b(int idx) {
    if (idx == NY / 2 * NX + NX / 3) return 10;
    if (idx == NY * 2 / 3 * NX + NX * 2 / 3) return -25;
    return 0;
}

__global__ void init_b(double* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        b[idx] = get_b(idx);
    }
}

__global__ void mul_mv_sub(double* res, const double* x, const double* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= SIZE) return;

    double tmp = -b[i];
    for (int j = 0; j < SIZE; ++j) {
        double a_ij = get_a(i, j);
        if (a_ij != 0.0) {
            tmp += a_ij * x[j];
        }
    }
    res[i] = tmp;
}

__global__ void update_x(double* x, const double* delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SIZE) {
        x[i] -= TAU * delta[i];
    }
}

__global__ void block_reduce_norm(const double* vec, double* block_results) {
    __shared__ typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage temp_storage;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double val = (i < SIZE) ? vec[i] * vec[i] : 0.0;

    double sum = cub::BlockReduce<double, BLOCK_SIZE>(temp_storage).Sum(val);

    if (threadIdx.x == 0) block_results[blockIdx.x] = sum;
}

double compute_norm(double* d_vec, double* d_temp, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block_reduce_norm<<<blocks, BLOCK_SIZE>>>(d_vec, d_temp);
    double* h_temp = new double[blocks];
    cudaMemcpy(h_temp, d_temp, blocks * sizeof(double), cudaMemcpyDeviceToHost);
    double total = 0.0;
    for (int i = 0; i < blocks; i++) total += h_temp[i];
    delete[] h_temp;
    return sqrt(total);
}

int main() {
    double *d_x, *d_b, *d_Axmb, *d_temp;
    cudaMalloc(&d_x, SIZE * sizeof(double));
    cudaMalloc(&d_b, SIZE * sizeof(double));
    cudaMalloc(&d_Axmb, SIZE * sizeof(double));
    cudaMalloc(&d_temp, ((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(double));

    cudaMemset(d_x, 0, SIZE * sizeof(double));
    init_b<<<(SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_b);

    int iterations = 0;
    double final_norm = 0.0;

    double norm_b = compute_norm(d_b, d_temp, SIZE);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    bool graph_created = false;

    auto start = high_resolution_clock::now();

    while (iterations < MAX_ITER) {
        if (!graph_created) {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            mul_mv_sub<<<(SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_Axmb, d_x, d_b);
            update_x<<<(SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_x, d_Axmb);

            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
            graph_created = true;
        }

        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);

        mul_mv_sub<<<(SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_Axmb, d_x, d_b);
        final_norm = compute_norm(d_Axmb, d_temp, SIZE);

        double rel_norm = final_norm / norm_b;
        printf("%lf >= %lf\r", rel_norm, EPS);
        fflush(stdout);
        if (rel_norm < EPS) break;

        iterations++;
    }
    
    printf("\33[2K\r");
    fflush(stdout);

    auto end = high_resolution_clock::now();
    duration<double> diff = end - start;

    double* h_x = new double[SIZE];
    cudaMemcpy(h_x, d_x, SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Matrix %dx%d processing time: %.4f sec. (Iterations: %d/%d, Final norm: %.8lf < %.8lf).\n",
           NX, NY, diff.count(), iterations, MAX_ITER, final_norm / norm_b, EPS);

    if ((NX == 10 && NY == 10) || (NX == 13 && NY == 13)) {
        cout << "Result matrix:\n";
        for (int i = 0; i < NY; ++i) {
            for (int j = 0; j < NX; ++j) {
                cout << fixed << setw(10) << setprecision(4) << h_x[i * NX + j] << " ";
            }
            cout << endl;
        }
    }

    FILE* f = fopen(OUT_FILE, "wb");
    fwrite(h_x, sizeof(double), SIZE, f);
    fclose(f);
    printf("Result matrix saved to file '%s'.\n", OUT_FILE);

    delete[] h_x;

    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_Axmb);
    cudaFree(d_temp);
    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    return 0;
}
