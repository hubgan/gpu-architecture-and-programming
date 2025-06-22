#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 1024
#define RADIUS 4
#define LENGTH 1024

__global__ void filter(const float* a, float* b, int length) {
	__shared__ float memory[BLOCK_SIZE + 2 * RADIUS];

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int localIndex = threadIdx.x + RADIUS;

	memory[localIndex] = a[threadIndex];

	if (threadIdx.x < RADIUS) {
		if (threadIndex - RADIUS < 0) {
			memory[localIndex - RADIUS] = a[length + threadIndex - RADIUS];
		}
		else {
			memory[localIndex - RADIUS] = a[threadIndex - RADIUS];
		}

		if (threadIndex + BLOCK_SIZE >= LENGTH) {
			memory[localIndex + BLOCK_SIZE] = a[threadIndex + BLOCK_SIZE - length];
		}
		else {
			memory[localIndex + BLOCK_SIZE] = a[threadIndex + BLOCK_SIZE];
		}
	}

	__syncthreads();

	float result = 0.0;

	for (int offset = -RADIUS; offset <= RADIUS; offset++) {
		result += memory[localIndex + offset];
	}

	b[threadIndex] = result;
}

int main() {
	float* x = (float*)malloc(sizeof(float) * LENGTH);
	float* y = (float*)malloc(sizeof(float) * LENGTH);

	for (int i = 0; i < LENGTH; ++i) {
		x[i] = 1.0f;
		y[i] = -1.0f;
	}

	float* d_x, * d_y;
	cudaMalloc(&d_x, sizeof(float) * LENGTH);
	cudaMalloc(&d_y, sizeof(float) * LENGTH);

	cudaMemcpy(d_x, x, sizeof(float) * LENGTH, cudaMemcpyHostToDevice);

	int blocksNum = (LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE;

	filter<<< blocksNum, BLOCK_SIZE >>>(d_x, d_y, LENGTH);
	cudaDeviceSynchronize();

	cudaMemcpy(y, d_y, sizeof(float) * LENGTH, cudaMemcpyDeviceToHost);

	printf("Idx\tX[i]\t\tY[i]\n");
	for (int i = 0; i < LENGTH; ++i) {
		printf("%d\t%.2f\t->\t%.2f\n", i, x[i], y[i]);
	}

	free(x);
	free(y);
	cudaFree(d_x);
	cudaFree(d_y);

	return 0;
}