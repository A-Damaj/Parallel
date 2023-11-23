#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define TILE_WIDTH 15


__global__ void MatMulTiled(float* A, float* B, float* C, int width) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float sum = 0;

    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
        ds_A[ty][tx] = A[Row * width + ph * TILE_WIDTH + tx];
        ds_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * width + Col];
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            sum += ds_A[ty][i] * ds_B[i][tx];
        __syncthreads();
    }

    C[Row * width + Col] = sum;
}


float randomFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
}


void generateRandomMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = randomFloat();
    }
}

int main() {
    // Matrix size
    int N;

    // Input matrix dimension N*N
    printf("Enter the dimensions of the matrices (N): ");
    scanf("%d", &N);

    // Allocate host memory
    float* h_A = (float*)malloc(N * N * sizeof(float));
    float* h_B = (float*)malloc(N * N * sizeof(float));
    float* h_C = (float*)malloc(N * N * sizeof(float));

    // Initialize host matrices with random numbers
    generateRandomMatrix(h_A, N);
    generateRandomMatrix(h_B, N);

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke the kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);

    // Measure the start time
     clock_t start_time = clock();

    MatMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Measure the end time
    clock_t end_time = clock();
    
    // Copy the result matrix C back to the host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Output the time for computation
    printf("Time taken for computation: %f seconds\n", time_taken);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
