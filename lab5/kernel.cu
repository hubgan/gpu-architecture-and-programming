
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <locale>

constexpr int BLOCK_SIZE = 1024;
constexpr int LENGTH = 16 * 1024 * 1024;

#define cudaCheck(err) if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
}

__global__ void reduce0(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2 * s)) == 0) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce1(int* g_idata, int* g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }

        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce3(int* g_idata, int* g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, int tid) {
    if (blockSize >= 64) {
        sdata[tid] += sdata[tid + 32];
    }

    if (blockSize >= 32) {
        sdata[tid] += sdata[tid + 16];
    }

    if (blockSize >= 16) {
        sdata[tid] += sdata[tid + 8];
    }

    if (blockSize >= 8) {
        sdata[tid] += sdata[tid + 4];
    }

    if (blockSize >= 4) {
        sdata[tid] += sdata[tid + 2];
    }

    if (blockSize >= 2) {
        sdata[tid] += sdata[tid + 1];
    }
}

__global__ void reduce6(int* g_idata, int* g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
    __syncthreads();

    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }

        __syncthreads();
    }

    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }

        __syncthreads();
    }

    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }

        __syncthreads();
    }

    if (tid < 32) {
        warpReduce<BLOCK_SIZE>(sdata, tid);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main() {
    setlocale(LC_CTYPE, "Polish");
    float time_0 = 0, time_1 = 0, time_3 = 0, time_6 = 0;
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    int* x = (int*)malloc(LENGTH * sizeof(int));
    int* y = (int*)malloc(LENGTH * sizeof(int));
    int* z = (int*)malloc(LENGTH * sizeof(int));

    int* d_x, * d_y, * d_z;
    cudaCheck(cudaMalloc(&d_x, LENGTH * sizeof(int)));
    cudaCheck(cudaMalloc(&d_y, LENGTH * sizeof(int)));
    cudaCheck(cudaMalloc(&d_z, LENGTH * sizeof(int)));

    const int blocks_amount = (LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int shared_amount = BLOCK_SIZE * sizeof(int);

    auto testReduce = [&](auto kernel, const char* label, int val, float& time) {
        for (int i = 0; i < LENGTH; i++) {
            x[i] = val;
        }

        cudaMemcpy(d_x, x, LENGTH * sizeof(int), cudaMemcpyHostToDevice);
        printf("<<<%d, %d>>> %s\n", blocks_amount, BLOCK_SIZE, label);

        cudaEventRecord(start);
        kernel << < blocks_amount, BLOCK_SIZE, shared_amount >> > (d_x, d_y, LENGTH);
        kernel << < 1, BLOCK_SIZE, shared_amount >> > (d_y, d_z, LENGTH);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cudaCheck(cudaGetLastError());

        cudaMemcpy(z, d_z, LENGTH * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Wynik końcowy" << std::endl;
        std::cout << "Suma: " << z[0] << std::endl;
        std::cout << label << " zajęło: " << time << " ms" << std::endl;
    };

    testReduce(reduce0, "REDUCE 0", 1, time_0);
    testReduce(reduce1, "REDUCE 1", 1, time_1);
    testReduce(reduce3, "REDUCE 3", 1, time_3);
    testReduce(reduce6, "REDUCE 6", 1, time_6);

    return 0;
}
