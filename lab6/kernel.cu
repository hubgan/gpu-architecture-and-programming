#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__device__ void partition(int* array, int left, int right, int** leftPtr, int** rightPtr, int* pivotValue) {
    while (left <= right) {
        while (array[left] < *pivotValue) {
            left++;
        }

        while (array[right] > *pivotValue) {
            right--;
        }

        if (left <= right) {
            int temp = array[left];
            array[left] = array[right];
            array[right] = temp;
            left++;
            right--;
        }
    }

    *leftPtr = array + left;
    *rightPtr = array + right;
}

__global__ void gpuQuickSort(int* array, int left, int right)
{
    int pivotValue = array[(left + right) / 2];
    int* leftPtr = array + left;
    int* rightPtr = array + right;

    partition(array, left, right, &leftPtr, &rightPtr, &pivotValue);

    int newRightIndex = rightPtr - array;
    int newLeftIndex = leftPtr - array;

    cudaStream_t leftStream, rightStream;
    cudaStreamCreateWithFlags(&leftStream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&rightStream, cudaStreamNonBlocking);

    if (left < newRightIndex) {
        gpuQuickSort << <1, 1, 0, leftStream >> > (array, left, newRightIndex);
    }

    if (newLeftIndex < right) {
        gpuQuickSort << <1, 1, 0, rightStream >> > (array, newLeftIndex, right);
    }

    cudaStreamDestroy(leftStream);
    cudaStreamDestroy(rightStream);
}

int main() {
    srand((unsigned)time(NULL));

    printf("Wybierz tryb dzialania programu:\n");
    printf("1) Przyklad\n");
    printf("2) Random\n");
    printf("Twoj wybor (1 lub 2): ");
    int chosenMode = 0;
    if (scanf("%d", &chosenMode) != 1) {
        fprintf(stderr, "Blad wczytywania wyboru.\n");
        return -1;
    }

    if (chosenMode == 1) {
        int hostArray[] = { 7, 3, 5, 1, 9, 2, 8, 4, 6 };
        int arraySize = sizeof(hostArray) / sizeof(int);

        int* deviceArray = nullptr;
        cudaError_t allocResult = cudaMalloc((void**)&deviceArray, sizeof(int) * arraySize);
        if (allocResult != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(allocResult));
            return -1;
        }
        cudaMemcpy(deviceArray, hostArray, sizeof(int) * arraySize, cudaMemcpyHostToDevice);

        gpuQuickSort << <1, 1 >> > (deviceArray, 0, arraySize - 1);
        cudaDeviceSynchronize();

        cudaMemcpy(hostArray, deviceArray, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);

        printf("\nWariant 1: Hardkodowana tablica posortowana:\n");
        for (int i = 0; i < arraySize; i++) {
            printf("%d ", hostArray[i]);
        }
        printf("\n");

        cudaFree(deviceArray);
    }
    else if (chosenMode == 2) {
        printf("Podaj dlugosc tablicy: ");
        int arraySize = 0;
        if (scanf("%d", &arraySize) != 1 || arraySize <= 0) {
            fprintf(stderr, "Blad: niepoprawna dlugosc tablicy.\n");
            return -1;
        }

        int* hostArray = (int*)malloc(sizeof(int) * arraySize);
        if (hostArray == NULL) {
            fprintf(stderr, "malloc failed: nie mozna zaalokowac pamieci na hosta\n");
            return -1;
        }

        for (int i = 0; i < arraySize; i++) {
            hostArray[i] = (rand() % 20001) - 10000;
        }

        int* deviceArray = nullptr;
        cudaError_t allocResultDev = cudaMalloc((void**)&deviceArray, sizeof(int) * arraySize);
        if (allocResultDev != cudaSuccess) {
            fprintf(stderr, "cudaMalloc (device) failed: %s\n", cudaGetErrorString(allocResultDev));
            free(hostArray);
            return -1;
        }
        cudaMemcpy(deviceArray, hostArray, sizeof(int) * arraySize, cudaMemcpyHostToDevice);

        gpuQuickSort << <1, 1 >> > (deviceArray, 0, arraySize - 1);
        cudaDeviceSynchronize();

        cudaMemcpy(hostArray, deviceArray, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);

        printf("\nWariant 2: Losowo wygenerowana tablica (N = %d) posortowana:\n", arraySize);
        for (int i = 0; i < arraySize; i++) {
            printf("%d ", hostArray[i]);
            if ((i & 0x1F) == 31)
                printf("\n");
        }
        printf("\n");

        cudaFree(deviceArray);
        free(hostArray);
    }
    else {
        fprintf(stderr, "Nieprawidlowy wybor: %d. Wybierz 1 lub 2.\n", chosenMode);
        return -1;
    }

    return 0;
}
