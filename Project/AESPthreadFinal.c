#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <pthread.h>

#define AES_GCM_KEY_SIZE 16
#define AES_GCM_IV_SIZE 12
#define AES_GCM_TAG_SIZE 16
#define NUM_THREADS 4

// This struct will be passed to each thread
typedef struct {
    FILE *file;
    char outfile[50];
    long start;
    long end;
} ThreadData;

void *encrypt_chunk(void *arg) {
    ThreadData *data = (ThreadData *)arg;

    // Open the output file
    FILE *outfile = fopen(data->outfile, "wb");
    if (!outfile) {
        perror("Unable to open output file");
        return NULL;
    }

    // Initialize the encryption context
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        printf("Unable to create cipher context\n");
        return NULL;
    }

    // Initialize AES-GCM with 128-bit key
    if (EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, NULL, NULL) != 1) {
        printf("Unable to initialize cipher\n");
        return NULL;
    }

    // Set the key and IV
    unsigned char key[AES_GCM_KEY_SIZE] = "ThisIsASampleKey"; // key
    unsigned char iv[AES_GCM_IV_SIZE];
    if (RAND_bytes(iv, sizeof(iv)) != 1) {
        printf("Unable to generate IV\n");
        return NULL;
    }
    if (EVP_EncryptInit_ex(ctx, NULL, NULL, key, iv) != 1) {
        printf("Unable to set key/IV\n");
        return NULL;
    }

    // Write the IV to the output file
    fwrite(iv, 1, AES_GCM_IV_SIZE, outfile);

    // Encrypt the file
    unsigned char buffer[1024], outbuf[1024 + EVP_MAX_BLOCK_LENGTH];
    int outlen;
    fseek(data->file, data->start, SEEK_SET);
    long remaining = data->end - data->start;
    while (remaining > 0) {
        size_t count = fread(buffer, 1, sizeof(buffer), data->file);
        if (count == 0) break;
        if (EVP_EncryptUpdate(ctx, outbuf, &outlen, buffer, count) != 1) {
            printf("Encryption failed\n");
            return NULL;
        }
        fwrite(outbuf, 1, outlen, outfile);
        remaining -= count;
    }

    // Finalize encryption and get the tag
    unsigned char tag[AES_GCM_TAG_SIZE];
    if (EVP_EncryptFinal_ex(ctx, outbuf, &outlen) != 1 || EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, AES_GCM_TAG_SIZE, tag) != 1) {
        printf("Unable to finalize encryption or get tag\n");
        return NULL;
    }
    fwrite(outbuf, 1, outlen, outfile);
    fwrite(tag, 1, AES_GCM_TAG_SIZE, outfile);  // Write the tag to the output file

    // Clean up
    EVP_CIPHER_CTX_free(ctx);
    fclose(outfile);

    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    // Open the input file
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        perror("Unable to open file");
        return 1;
    }

    // Get the size of the file
    fseek(file, 0, SEEK_END);
    long filesize = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Create the threads
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].file = file;
        sprintf(thread_data[i].outfile, "encrypted_file_%d", i);
        thread_data[i].start = i * filesize / NUM_THREADS;
        thread_data[i].end = (i + 1) * filesize / NUM_THREADS;
        pthread_create(&threads[i], NULL, encrypt_chunk, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // Clean up
    fclose(file);

    return 0;
}
