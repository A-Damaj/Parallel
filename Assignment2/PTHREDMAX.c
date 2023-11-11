#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define MAX_THREADS 4

int N;  // Both matrices are now N*N
int** matrix1;
int** matrix2;
int** result;

void* multiply(void* arg) {
    long thread_id = (long)arg;

    int chunk_size = N / MAX_THREADS;
    int start_row = thread_id * chunk_size;
    int end_row = (thread_id == MAX_THREADS - 1) ? N : (thread_id + 1) * chunk_size;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = 0;
            for (int k = 0; k < N; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return NULL;
}

int main() {
    printf("Enter the dimensions of the matrices (N): ");
    scanf("%d", &N);

    srand(time(NULL)); 

    matrix1 = (int**)malloc(N * sizeof(int*));
    matrix2 = (int**)malloc(N * sizeof(int*));
    result = (int**)malloc(N * sizeof(int*));

    for (int i = 0; i < N; i++) {
        matrix1[i] = (int*)malloc(N * sizeof(int));
        matrix2[i] = (int*)malloc(N * sizeof(int));
        result[i] = (int*)malloc(N * sizeof(int));
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix1[i][j] = rand() % 10;
            matrix2[i][j] = rand() % 10;
        }
    }

    clock_t start_time = clock();

    pthread_t threads[MAX_THREADS];

    for (long i = 0; i < MAX_THREADS; i++) {
        pthread_create(&threads[i], NULL, multiply, (void*)i);
    }

    for (long i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_t end_time = clock();

    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Time taken for matrix multiplication: %f seconds\n", time_taken);

    for (int i = 0; i < N; i++) {
        free(matrix1[i]);
        free(matrix2[i]);
        free(result[i]);
    }
    free(matrix1);
    free(matrix2);
    free(result);

    return 0;
}

