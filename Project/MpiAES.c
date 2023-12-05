#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <mpi.h>

#define AES_GCM_KEY_SIZE 16
#define AES_GCM_IV_SIZE 12
#define AES_GCM_TAG_SIZE 16

void encrypt_chunk(FILE *file, const char *outfile, long start, long end) {
    // Open the output file
    FILE *out = fopen(outfile, "wb");
    if (!out) {
        perror("Unable to open output file");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Initialize the encryption context
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        printf("Unable to create cipher context\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Initialize AES-GCM with 128-bit key
    if (EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, NULL, NULL) != 1) {
        printf("Unable to initialize cipher\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Set the key and IV
    unsigned char key[AES_GCM_KEY_SIZE] = "ThisIsASampleKey"; // key
    unsigned char iv[AES_GCM_IV_SIZE];
    if (RAND_bytes(iv, sizeof(iv)) != 1) {
        printf("Unable to generate IV\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (EVP_EncryptInit_ex(ctx, NULL, NULL, key, iv) != 1) {
        printf("Unable to set key/IV\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Write the IV to the output file
    fwrite(iv, 1, AES_GCM_IV_SIZE, out);

    // Encrypt the file
    unsigned char buffer[1024], outbuf[1024 + EVP_MAX_BLOCK_LENGTH];
    int outlen;
    fseek(file, start, SEEK_SET);
    long remaining = end - start;
    while (remaining > 0) {
        size_t count = fread(buffer, 1, sizeof(buffer), file);
        if (count == 0) break;
        if (EVP_EncryptUpdate(ctx, outbuf, &outlen, buffer, count) != 1) {
            printf("Encryption failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fwrite(outbuf, 1, outlen, out);
        remaining -= count;
    }

    // Finalize encryption and get the tag
    unsigned char tag[AES_GCM_TAG_SIZE];
    if (EVP_EncryptFinal_ex(ctx, outbuf, &outlen) != 1 || EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, AES_GCM_TAG_SIZE, tag) != 1) {
        printf("Unable to finalize encryption or get tag\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    fwrite(outbuf, 1, outlen, out);
    fwrite(tag, 1, AES_GCM_TAG_SIZE, out);  // Write the tag to the output file

    // Clean up
    EVP_CIPHER_CTX_free(ctx);
    fclose(out);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Open the input file
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        perror("Unable to open file");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Get the size of the file
    fseek(file, 0, SEEK_END);
    long filesize = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Calculate the start and end positions for this process
    long start = rank * filesize / size;
    long end = (rank + 1) * filesize / size;

    // Generate the output filename
    char outfile[50];
    sprintf(outfile, "encrypted_file_%d", rank);

    // Encrypt the chunk of the file
    encrypt_chunk(file, outfile, start, end);

    // Clean up
    fclose(file);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
