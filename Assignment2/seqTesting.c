#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    int M, N;

  
    printf("Enter the dimensions of the matrices (M N): ");
    scanf("%d %d", &M, &N);

    srand(time(NULL));

 
    int** matrix1 = allocateMatrix(M, N);
    int** matrix2 = allocateMatrix(N, M); 

    generateRandomMatrix(matrix1, M, N);
    generateRandomMatrix(matrix2, N, M);

    clock_t start_time = clock();

    int** result = multiplyMatrices(matrix1, M, N, matrix2, N, M);

    // Measure the end time
    clock_t end_time = clock();

    if (result != NULL) {
        printf("Time taken for matrix multiplication: %f seconds\n", ((double)(end_time - start_time)) / CLOCKS_PER_SEC);

        freeMatrix(matrix1, M);
        freeMatrix(matrix2, N);
        freeMatrix(result, M);
    }

    return 0;
}

