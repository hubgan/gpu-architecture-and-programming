#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <locale>

#define TILE_WIDTH 2

__global__ void matrixMulShared(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        if (Row < M && (ph * TILE_WIDTH + tx) < K) {
            ds_A[ty][tx] = A[Row * K + ph * TILE_WIDTH + tx];
        }
        else {
            ds_A[ty][tx] = 0.0f;
        }

        if (Col < N && (ph * TILE_WIDTH + ty) < K) {
            ds_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + Col];
        }
        else {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    if (Row < M && Col < N) {
        C[Row * N + Col] = Pvalue;
    }
}

void fillMatrixRandom(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand() % 10);
    }
}

void fillMatrixManual(float* mat, int rows, int cols, const char* name) {
    std::cout << "Podaj wartosci macierzy " << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << name << "[" << i << "][" << j << "]: ";
            std::cin >> mat[i * cols + j];
        }
    }
}

void printMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    setlocale(LC_CTYPE, "Polish");
    srand(static_cast<unsigned int>(time(0)));

    int M, K, N;
    char mode;

    std::cout << "Wybierz tryb ładowania macierzy:\n";
    std::cout << "a - automatyczny (losowy)\n";
    std::cout << "r - ręczny\n";
    std::cout << "p - przykład\n";
    std::cout << "Twój wybór: ";
    std::cin >> mode;
    mode = std::tolower(mode);

    float* h_A;
    float* h_B;
    float* h_C;

    if (mode == 'p') {
        M = 2;
        K = 3;
        N = 2;
        h_A = (float*)malloc(M * K * sizeof(float));
        h_B = (float*)malloc(K * N * sizeof(float));
        h_C = (float*)malloc(M * N * sizeof(float));

        float example_A[] = {
            1, 2, 3,
            4, 5, 6
        };
        float example_B[] = {
            7, 8,
            9, 10,
            11, 12
        };

        std::copy(example_A, example_A + M * K, h_A);
        std::copy(example_B, example_B + K * N, h_B);
    }
    else {
        std::cout << "Podaj wymiary macierzy: M (wiersze A), K (kolumny A / wiersze B), N (kolumny B): ";
        std::cin >> M >> K >> N;

        h_A = (float*)malloc(M * K * sizeof(float));
        h_B = (float*)malloc(K * N * sizeof(float));
        h_C = (float*)malloc(M * N * sizeof(float));

        if (mode == 'a') {
            fillMatrixRandom(h_A, M, K);
            fillMatrixRandom(h_B, K, N);
        }
        else if (mode == 'r') {
            fillMatrixManual(h_A, M, K, "A");
            fillMatrixManual(h_B, K, N, "B");
        }
        else {
            std::cerr << "Niepoprawny tryb. Zakończenie.\n";
            return 1;
        }
    }

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    matrixMulShared << <dimGrid, dimBlock >> > (d_A, d_B, d_C, M, K, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nMacierz A:\n";
    printMatrix(h_A, M, K);

    std::cout << "\nMacierz B:\n";
    printMatrix(h_B, K, N);

    std::cout << "\nMacierz C = A * B:\n";
    printMatrix(h_C, M, N);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << "\nCzas wykonania kernela (shared memory): " << elapsed << " ms\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

//// Wersja 3 - Mnożenie macierzy kwadratowych z pamięcią współdzieloną
//#define TILE_WIDTH 2
//
//__global__ void matrixMulShared(float* A, float* B, float* C, int Width) {
//    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
//    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
//
//    int bx = blockIdx.x; int by = blockIdx.y;
//    int tx = threadIdx.x; int ty = threadIdx.y;
//
//    int Row = by * TILE_WIDTH + ty;
//    int Col = bx * TILE_WIDTH + tx;
//
//    float Pvalue = 0.0f;
//
//    for (int ph = 0; ph < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
//        if (Row < Width && ph * TILE_WIDTH + tx < Width) {
//            ds_A[ty][tx] = A[Row * Width + ph * TILE_WIDTH + tx];
//        }
//        else {
//            ds_A[ty][tx] = 0.0f;
//        }
//           
//
//        if (Col < Width && ph * TILE_WIDTH + ty < Width) {
//            ds_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * Width + Col];
//        }   
//        else {
//            ds_B[ty][tx] = 0.0f;
//        }
//            
//        __syncthreads();
//
//        for (int k = 0; k < TILE_WIDTH; ++k) {
//            Pvalue += ds_A[ty][k] * ds_B[k][tx];
//        }
//           
//        __syncthreads();
//    }
//
//    if (Row < Width && Col < Width) {
//        C[Row * Width + Col] = Pvalue;
//    }
//}
//
//void fillMatrix(float* mat, int Width) {
//    for (int i = 0; i < Width * Width; ++i) {
//        mat[i] = static_cast<float>(rand() % 10);
//    }
//}
//
//void printMatrix(float* mat, int Width) {
//    for (int i = 0; i < Width; ++i) {
//        for (int j = 0; j < Width; ++j) {
//            std::cout << mat[i * Width + j] << "\t";
//        }
//        std::cout << std::endl;
//    }
//}
//
//int main() {
//    setlocale(LC_CTYPE, "Polish");
//    srand(static_cast<unsigned int>(time(0)));
//
//    int Width;
//    std::cout << "Podaj rozmiar macierzy (np. 256, 512, 1024): ";
//    std::cin >> Width;
//
//    int size = Width * Width;
//    int byteSize = size * sizeof(float);
//
//    float* h_A = (float*)malloc(byteSize);
//    float* h_B = (float*)malloc(byteSize);
//    float* h_C = (float*)malloc(byteSize);
//
//    fillMatrix(h_A, Width);
//    fillMatrix(h_B, Width);
//
//    float* d_A, * d_B, * d_C;
//    cudaMalloc(&d_A, byteSize);
//    cudaMalloc(&d_B, byteSize);
//    cudaMalloc(&d_C, byteSize);
//
//    cudaMemcpy(d_A, h_A, byteSize, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_B, h_B, byteSize, cudaMemcpyHostToDevice);
//
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);
//
//    dim3 dimGrid((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH);
//    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
//
//    matrixMulShared << <dimGrid, dimBlock >> > (d_A, d_B, d_C, Width);
//
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//
//    cudaMemcpy(h_C, d_C, byteSize, cudaMemcpyDeviceToHost);
//
//    std::cout << "\nMacierz A:\n"; printMatrix(h_A, Width);
//    std::cout << "\nMacierz B:\n"; printMatrix(h_B, Width);
//    std::cout << "\nMacierz C (wynik):\n"; printMatrix(h_C, Width);
//
//    float elapsed;
//    cudaEventElapsedTime(&elapsed, start, stop);
//    std::cout << "\nCzas wykonania kernela (z pamięcią współdzieloną): " << elapsed << " ms\n";
//
//    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
//    free(h_A); free(h_B); free(h_C);
//    cudaEventDestroy(start); cudaEventDestroy(stop);
//
//    return 0;
//}

//// Wersja 2 - Mnożenie macierzy kwadratowej wykorzystująca wiele bloków
//#define TILE_WIDTH 2
//
//__global__ void matrixMulKernel(float* A, float* B, float* C, int Width) {
//    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
//    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
//
//    float sum = 0.0f;
//    if (row < Width && col < Width) {
//        for (int k = 0; k < Width; ++k) {
//            sum += A[row * Width + k] * B[k * Width + col];
//        }
//        C[row * Width + col] = sum;
//    }
//}
//
//void fillMatrix(float* mat, int Width) {
//    for (int i = 0; i < Width * Width; ++i) {
//        mat[i] = static_cast<float>(rand() % 10);
//    }
//}
//
//void printMatrix(float* mat, int Width) {
//    for (int i = 0; i < Width; ++i) {
//        for (int j = 0; j < Width; ++j) {
//            std::cout << mat[i * Width + j] << "\t";
//        }
//        std::cout << std::endl;
//    }
//}
//
//int main() {
//    setlocale(LC_CTYPE, "Polish");
//    srand(static_cast<unsigned int>(time(0)));
//
//    int Width;
//    std::cout << "Podaj rozmiar macierzy (np. 256, 512, 1024): ";
//    std::cin >> Width;
//
//    int size = Width * Width;
//    int byteSize = size * sizeof(float);
//
//    float* h_A = (float*)malloc(byteSize);
//    float* h_B = (float*)malloc(byteSize);
//    float* h_C = (float*)malloc(byteSize);
//
//    fillMatrix(h_A, Width);
//    fillMatrix(h_B, Width);
//
//    float* d_A, * d_B, * d_C;
//    cudaMalloc(&d_A, byteSize);
//    cudaMalloc(&d_B, byteSize);
//    cudaMalloc(&d_C, byteSize);
//
//    cudaMemcpy(d_A, h_A, byteSize, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_B, h_B, byteSize, cudaMemcpyHostToDevice);
//
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);
//
//    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
//    dim3 dimGrid((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH);
//
//    matrixMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C, Width);
//
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//
//    cudaMemcpy(h_C, d_C, byteSize, cudaMemcpyDeviceToHost);
//
//    std::cout << "\nMacierz A:\n"; printMatrix(h_A, Width);
//    std::cout << "\nMacierz B:\n"; printMatrix(h_B, Width);
//    std::cout << "\nMacierz C (wynik):\n"; printMatrix(h_C, Width);
//
//    float elapsed;
//    cudaEventElapsedTime(&elapsed, start, stop);
//    std::cout << "\nCzas wykonania kernela: " << elapsed << " ms\n";
//
//    cudaFree(d_A);
//    cudaFree(d_B);
//    cudaFree(d_C);
//    free(h_A);
//    free(h_B);
//    free(h_C);
//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);
//
//    return 0;
//}

//// Wersja 1 - Mnożenie macierzy kwadratowej wykorzystując jeden blok
//#define MAX_WIDTH 32  // Maksymalna wielkość (bo jeden blok = 1024 wątki)
//
//__global__ void mul(float* Md, float* Nd, float* Pd, int Width) {
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//    float w = 0.0f;
//    for (int k = 0; k < Width; ++k) {
//        w += Md[ty * Width + k] * Nd[k * Width + tx];
//    }
//    Pd[ty * Width + tx] = w;
//}
//
//void fillMatrix(float* mat, int Width) {
//    for (int i = 0; i < Width * Width; ++i) {
//        mat[i] = static_cast<float>(rand() % 10);
//    }
//}
//
//void printMatrix(float* mat, int Width) {
//    for (int i = 0; i < Width; ++i) {
//        for (int j = 0; j < Width; ++j) {
//            std::cout << mat[i * Width + j] << "\t";
//        }
//        std::cout << std::endl;
//    }
//}
//
//int main() {
//    setlocale(LC_CTYPE, "Polish");
//    srand(static_cast<unsigned int>(time(0)));
//
//    int Width;
//    std::cout << "Podaj rozmiar macierzy (maks. " << MAX_WIDTH << "): ";
//    std::cin >> Width;
//
//    if (Width <= 0 || Width > MAX_WIDTH) {
//        std::cerr << "Nieprawidłowy rozmiar. Dozwolony zakres: 1 - " << MAX_WIDTH << std::endl;
//        return 1;
//    }
//
//    int size = Width * Width;
//    int byteSize = size * sizeof(float);
//
//    float* h_A = (float*)malloc(byteSize);
//    float* h_B = (float*)malloc(byteSize);
//    float* h_C = (float*)malloc(byteSize);
//
//    fillMatrix(h_A, Width);
//    fillMatrix(h_B, Width);
//
//    float* d_A, * d_B, * d_C;
//    cudaMalloc(&d_A, byteSize);
//    cudaMalloc(&d_B, byteSize);
//    cudaMalloc(&d_C, byteSize);
//
//    cudaMemcpy(d_A, h_A, byteSize, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_B, h_B, byteSize, cudaMemcpyHostToDevice);
//
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);
//
//    dim3 threadsPerBlock(Width, Width);
//    mul << <1, threadsPerBlock >> > (d_A, d_B, d_C, Width);
//
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//
//    cudaMemcpy(h_C, d_C, byteSize, cudaMemcpyDeviceToHost);
//
//    std::cout << "\nMacierz A:\n";
//    printMatrix(h_A, Width);
//    std::cout << "\nMacierz B:\n";
//    printMatrix(h_B, Width);
//    std::cout << "\nMacierz C (wynik):\n";
//    printMatrix(h_C, Width);
//
//    float elapsed = 0;
//    cudaEventElapsedTime(&elapsed, start, stop);
//    std::cout << "\nCzas wykonania kernela: " << elapsed << " ms\n";
//
//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);
//    cudaFree(d_A);
//    cudaFree(d_B);
//    cudaFree(d_C);
//    free(h_A);
//    free(h_B);
//    free(h_C);
//
//    return 0;
//}
