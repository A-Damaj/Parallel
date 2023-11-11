#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int randomInt(int min, int max) {
    return min + rand() % (max - min + 1);
}

int** allocateMatrix(int rows, int cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
    }
    return matrix;
}

void generateRandomMatrix(int** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = randomInt(1, 10); 
        }
    }
}

void freeMatrix(int** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int** multiplyMatrices(int** matrix1, int rows1, int cols1, int** matrix2, int rows2, int cols2) {
    if (cols1 != rows2) {
        printf("Matrix multiplication is not possible with these dimensions.\n");
        return NULL;
    }

    int** result = allocateMatrix(rows1, cols2);

    #pragma omp parallel for
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < cols1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

int main() {
    int N;
     omp_set_num_threads(4);
    printf("Enter the dimension of the matrices (N): ");
    scanf("%d", &N);

    srand(time(NULL)); 

    int** matrix1 = allocateMatrix(N, N);
    int** matrix2 = allocateMatrix(N, N);

    generateRandomMatrix(matrix1, N, N);
    generateRandomMatrix(matrix2, N, N);

    double start_time = omp_get_wtime();

    int** result = multiplyMatrices(matrix1, N, N, matrix2, N, N);

    // Measure the end time
    double end_time = omp_get_wtime();

    if (result != NULL) {
        printf("Time taken for matrix multiplication: %f seconds\n", end_time - start_time);

        freeMatrix(matrix1, N);
        freeMatrix(matrix2, N);
        freeMatrix(result, N);
    }

    return 0;
}

