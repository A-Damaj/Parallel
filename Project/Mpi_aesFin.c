#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <openssl/aes.h>

#define N 16 // Size of the array
//mpicc -o mpi_aes mpi_aes.c -lssl -lcrypto
//mpirun -n <number_of_processes> ./mpi_aes

void encryptAES(unsigned char *input, unsigned char *key, unsigned char *output) {
    AES_KEY aesKey;
    AES_set_encrypt_key(key, 128, &aesKey);
    AES_encrypt(input, output, &aesKey);
}

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned char *data;
    unsigned char key[16] = "0123456789ABCDEF"; // 128-bit key

    if (rank == 0) {
        // Master process generates the array
        data = (unsigned char *)malloc(N * sizeof(unsigned char));
        for (int i = 0; i < N; i++) {
            data[i] = (unsigned char)(i % 256);
        }

        // Master broadcasts the array to all other processes
        MPI_Bcast(data, N, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    } else {
        // Worker processes receive the array broadcasted by the master
        data = (unsigned char *)malloc(N * sizeof(unsigned char));
        MPI_Bcast(data, N, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

    // Calculate chunk size for each process
    int chunkSize = N / size;
    int start = rank * chunkSize;
    int end = (rank + 1) * chunkSize;

    // Encryption loop
    for (int i = start; i < end; i++) {
        encryptAES(&data[i], key, &data[i]);
    }

    // Gather results at the master process
    if (rank == 0) {
        unsigned char *encryptedData = (unsigned char *)malloc(N * sizeof(unsigned char));
        MPI_Gather(data + start, chunkSize, MPI_UNSIGNED_CHAR, encryptedData, chunkSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        // Display the encrypted array
        printf("Encrypted Array: ");
        for (int i = 0; i < N; i++) {
            printf("%02x ", encryptedData[i]);
        }
        printf("\n");

        free(encryptedData);
    } else {
        // Worker processes send their encrypted data to the master
        MPI_Gather(data + start, chunkSize, MPI_UNSIGNED_CHAR, NULL, 0, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

    free(data);

    MPI_Finalize();
    return 0;
}
