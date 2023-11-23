#include <stdio.h>
#include <stdlib.h>
#include <time.h>


__global__ void MatMulSimple(float* A, float* B, float* C, int width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < width && Col < width) {
        float sum = 0.0;
        for (int k = 0; k < width; ++k) {
            sum += A[Row * width + k] * B[k * width + Col];
        }
        C[Row * width + Col] = sum;
    }
}

int main() {

    int width;
    printf("Enter the dimensions of the matrices (N): ");
    scanf("%d", &width);


    float* h_A = (float*)malloc(width * width * sizeof(float));
    float* h_B = (float*)malloc(width * width * sizeof(float));
    float* h_C = (float*)malloc(width * width * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < width * width; ++i) {
        h_A[i] = (float)(rand() % 10);
        h_B[i] = (float)(rand() % 10);
    }

    float* d_A;
    cudaMalloc(&d_A, width * width * sizeof(float));
    float* d_B;
    cudaMalloc(&d_B, width * width * sizeof(float));
    float* d_C;
    cudaMalloc(&d_C, width * width * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, width * width * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y);

    // Measure the start time
    clock_t start_time = clock();

    // Invoke the kernel
    MatMulSimple<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);

    // Measure the end time
    clock_t end_time = clock();

    // Copy the result matrix C back to the host
    cudaMemcpy(h_C, d_C, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate the time taken for computation
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
