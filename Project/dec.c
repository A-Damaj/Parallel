#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>

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

    // Initialize the decryption context
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        printf("Unable to create cipher context\n");
        return 1;
    }

    // Initialize AES-GCM with 128-bit key
    if (EVP_DecryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, NULL, NULL) != 1) {
        printf("Unable to initialize cipher\n");
        return 1;
    }

    // Set the key and IV
    unsigned char key[AES_GCM_KEY_SIZE] = "ThisIsASampleKey"; // Replace with your key
    unsigned char iv[AES_GCM_IV_SIZE];
    if (fread(iv, 1, AES_GCM_IV_SIZE, file) != AES_GCM_IV_SIZE) {
        printf("Unable to read IV\n");
        return 1;
    }
    if (EVP_DecryptInit_ex(ctx, NULL, NULL, key, iv) != 1) {
        printf("Unable to set key/IV\n");
        return 1;
    }

    // Open the output file
    FILE *outfile = fopen("decrypted_file", "wb");
    if (!outfile) {
        perror("Unable to open output file");
        return 1;
    }

    // Decrypt the file
    unsigned char buffer[1024], outbuf[1024 + EVP_MAX_BLOCK_LENGTH];
    int outlen;
    while (1) {
        size_t count = fread(buffer, 1, sizeof(buffer), file);
        if (count == 0) break;
        if (EVP_DecryptUpdate(ctx, outbuf, &outlen, buffer, count) != 1) {
            printf("Decryption failed\n");
            return 1;
        }
        fwrite(outbuf, 1, outlen, outfile);
    }

    // Read the tag from the file
    unsigned char tag[AES_GCM_TAG_SIZE];
    if (fread(tag, 1, AES_GCM_TAG_SIZE, file) != AES_GCM_TAG_SIZE) {
        printf("Unable to read tag\n");
        return 1;
    }

    // Set the expected tag value
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, AES_GCM_TAG_SIZE, tag) != 1) {
        printf("Unable to set expected tag value\n");
        return 1;
    }

    // Finalize decryption
    if (EVP_DecryptFinal_ex(ctx, outbuf, &outlen) <= 0) {
        printf("Tag check failed\n");
        return 1;
    }
    fwrite(outbuf, 1, outlen, outfile);

    // Clean up
    EVP_CIPHER_CTX_free(ctx);
    fclose(file);
    fclose(outfile);

    return 0;
}
