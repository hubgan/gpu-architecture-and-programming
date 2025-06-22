#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <locale>

#define TPB 1024 // THREADS_PER_BLOCK

__global__ void addMatrices(int* a, int* b, int* c, int totalSize) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < totalSize) {
		c[index] = a[index] + b[index];
	}
}

void fillMatrixRandom(int* matrix, int size) {
	for (int i = 0; i < size; ++i) {
		matrix[i] = rand() % 100;
	}
}

void fillMatrixManual(int* matrix, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			std::cout << "Element [" << i << "][" << j << "]: ";
			std::cin >> matrix[i * cols + j];
		}
	}
}

void printMatrix(int* matrix, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			std::cout << matrix[i * cols + j] << "\t";
		}
		std::cout << std::endl;
	}
}

int main() {
	setlocale(LC_CTYPE, "Polish");
	srand(time(NULL));

	int rows, cols;
	std::cout << "Podaj liczbe wierszy: ";
	std::cin >> rows;
	std::cout << "Podaj liczbe kolumn: ";
	std::cin >> cols;

	int size = rows * cols;
	int byteSize = size * sizeof(int);

	int* h_a = (int*)malloc(byteSize);
	int* h_b = (int*)malloc(byteSize);
	int* h_c = (int*)malloc(byteSize);

	char mode;
	std::cout << "Wprowadź sposób wprowadzania danych. Ręcznie (r) lub Automatycznie (a)";
	std::cin >> mode;

	if (mode == 'r' || mode == 'R') {
		std::cout << "Wprowadź macierz A:" << std::endl;
		fillMatrixManual(h_a, rows, cols);
		std::cout << "Wprowadź macierz B:" << std::endl;
		fillMatrixManual(h_b, rows, cols);
	}
	else {
		fillMatrixRandom(h_a, size);
		fillMatrixRandom(h_b, size);
	}

	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, byteSize);
	cudaMalloc(&d_b, byteSize);
	cudaMalloc(&d_c, byteSize);

	cudaMemcpy(d_a, h_a, byteSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, byteSize, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int blocksPerGrid = (size + TPB - 1) / TPB;
	addMatrices << < blocksPerGrid, TPB >> > (d_a, d_b, d_c, size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaMemcpy(h_c, d_c, byteSize, cudaMemcpyDeviceToHost);

	std::cout << std::endl << "Macierz A" << std::endl;
	printMatrix(h_a, rows, cols);
	std::cout << std::endl << "Macierz B" << std::endl;
	printMatrix(h_b, rows, cols);
	std::cout << std::endl << "Macierz C" << std::endl;
	printMatrix(h_c, rows, cols);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << std::endl << "Czas wykonania kernela: " << milliseconds << " ms" << std::endl;

	free(h_a);
	free(h_b);
	free(h_c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
