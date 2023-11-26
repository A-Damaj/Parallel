#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#define AES_GCM_KEY_SIZE 16
#define AES_GCM_IV_SIZE 12
#define AES_GCM_TAG_SIZE 16

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

    // Initialize the encryption context
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        printf("Unable to create cipher context\n");
        return 1;
    }

    // Initialize AES-GCM with 128-bit key
    if (EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, NULL, NULL) != 1) {
        printf("Unable to initialize cipher\n");
        return 1;
    }

    // Set the key and IV
    unsigned char key[AES_GCM_KEY_SIZE] = "ThisIsASampleKey"; // key
    unsigned char iv[AES_GCM_IV_SIZE];
    if (RAND_bytes(iv, sizeof(iv)) != 1) {
        printf("Unable to generate IV\n");
        return 1;
    }
    if (EVP_EncryptInit_ex(ctx, NULL, NULL, key, iv) != 1) {
        printf("Unable to set key/IV\n");
        return 1;
    }

    // Open the output file
    FILE *outfile = fopen("encrypted_file", "wb");
    if (!outfile) {
        perror("Unable to open output file");
        return 1;
    }

    // Write the IV to the output file
    fwrite(iv, 1, AES_GCM_IV_SIZE, outfile);

    // Encrypt the file
    unsigned char buffer[1024], outbuf[1024 + EVP_MAX_BLOCK_LENGTH];
    int outlen;
    while (1) {
        size_t count = fread(buffer, 1, sizeof(buffer), file);
        if (count == 0) break;
        if (EVP_EncryptUpdate(ctx, outbuf, &outlen, buffer, count) != 1) {
            printf("Encryption failed\n");
            return 1;
        }
        fwrite(outbuf, 1, outlen, outfile);
    }

    // Finalize encryption and get the tag
    unsigned char tag[AES_GCM_TAG_SIZE];
    if (EVP_EncryptFinal_ex(ctx, outbuf, &outlen) != 1 || EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, AES_GCM_TAG_SIZE, tag) != 1) {
        printf("Unable to finalize encryption or get tag\n");
        return 1;
    }
    fwrite(outbuf, 1, outlen, outfile);
    fwrite(tag, 1, AES_GCM_TAG_SIZE, outfile);

    // Clean up
    EVP_CIPHER_CTX_free(ctx);
    fclose(file);
    fclose(outfile);

    return 0;
}
