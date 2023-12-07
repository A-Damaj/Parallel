#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <openssl/evp.h>

#define N 100000 // Size of the array

void handleErrors(void)
{
    ERR_print_errors_fp(stderr);
    abort();
}

int encryptAESGCM(unsigned char *plaintext, int plaintext_len, unsigned char *key,
            unsigned char *iv, unsigned char *ciphertext, unsigned char *tag)
{
    EVP_CIPHER_CTX *ctx;

    int len;

    int ciphertext_len;

    /* Create and initialise the context */
    if(!(ctx = EVP_CIPHER_CTX_new())) handleErrors();

    /* Initialise the encryption operation. */
    if(1 != EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, NULL, NULL))
        handleErrors();

    /* Set IV length if default 12 bytes (96 bits) is not appropriate */
    if(1 != EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 16, NULL))
        handleErrors();

    /* Initialise key and IV */
    if(1 != EVP_EncryptInit_ex(ctx, NULL, NULL, key, iv)) handleErrors();

    /* Provide the message to be encrypted, and obtain the encrypted output.
     * EVP_EncryptUpdate can be called multiple times if necessary
     */
    if(1 != EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len))
        handleErrors();
    ciphertext_len = len;

    /* Finalise the encryption. Normally ciphertext bytes may be written at
     * this stage, but this does not occur in GCM mode
     */
    if(1 != EVP_EncryptFinal_ex(ctx, ciphertext + len, &len)) handleErrors();
    ciphertext_len += len;

    /* Get the tag */
    if(1 != EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tag))
        handleErrors();

    /* Clean up */
    EVP_CIPHER_CTX_free(ctx);

    return ciphertext_len;
}

int main(int argc, char **argv) {
    int rank, size;
    double startTime, endTime;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int chunkSize = N / size;

    // Calculate chunk size for each process
    int start = rank * chunkSize;
    int end = (rank + 1) * chunkSize;

    unsigned char *data = (unsigned char *)malloc(N * sizeof(unsigned char));
    unsigned char key[16] = "0123456789ABCDEF"; // 128-bit key
    unsigned char iv[16]; // 128-bit IV
    unsigned char tag[16]; // 128-bit tag

    // Master process gathers the encrypted data from all processes
    if (rank == 0) {
        unsigned char *encryptedData = (unsigned char *)malloc(N * sizeof(unsigned char));
        startTime = MPI_Wtime(); // Start measuring time

        MPI_Gather(data, chunkSize, MPI_UNSIGNED_CHAR, encryptedData, chunkSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        endTime = MPI_Wtime(); // Stop measuring time

        printf("Time taken for encryption: %f seconds\n", endTime - startTime);

        free(encryptedData);
    } else {
        // Worker processes send their encrypted data to the master
        MPI_Gather(data + start, chunkSize, MPI_UNSIGNED_CHAR, NULL, 0, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

    // Encryption loop
    for (int i = start; i < end; i++) {
        encryptAESGCM(&data[i], 1, key, iv, &data[i], tag);
    }

    // Gather results at the master process
    if (rank == 0) {
        unsigned char *encryptedData = (unsigned char *)malloc(N * sizeof(unsigned char));
        startTime = MPI_Wtime(); // Start measuring time

        MPI_Gather(data + start, chunkSize, MPI_UNSIGNED_CHAR, encryptedData, chunkSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        endTime = MPI_Wtime(); // Stop measuring time

        printf("Time taken for encryption: %f seconds\n", endTime - startTime);

        free(encryptedData);
    } else {
        // Worker processes send their encrypted data to the master
        MPI_Gather(data + start, chunkSize, MPI_UNSIGNED_CHAR, NULL, 0, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

    free(data);

    MPI_Finalize();
    return 0;
}






