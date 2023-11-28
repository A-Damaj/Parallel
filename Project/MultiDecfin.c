#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#define AES_GCM_KEY_SIZE 16
#define AES_GCM_IV_SIZE 12
#define AES_GCM_TAG_SIZE 16
#define NUM_FILES 4

int main() {
    // Open the output file
    FILE *outfile = fopen("decrypted_file", "wb");
    if (!outfile) {
        perror("Unable to open output file");
        return 1;
    }

    for (int i = 0; i < NUM_FILES; i++) {
        char filename[50];
        sprintf(filename, "encrypted_file_%d", i);

        // Open the input file
        FILE *file = fopen(filename, "rb");
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
        unsigned char key[AES_GCM_KEY_SIZE] = "ThisIsASampleKey"; // key
        unsigned char iv[AES_GCM_IV_SIZE];
        fread(iv, 1, AES_GCM_IV_SIZE, file);  // Read a new IV for each chunk
        if (EVP_DecryptInit_ex(ctx, NULL, NULL, key, iv) != 1) {
            printf("Unable to set key/IV\n");
            return 1;
        }

        // Get the size of the file
        fseek(file, 0, SEEK_END);
        long filesize = ftell(file);
        fseek(file, 0, SEEK_SET);

        // Read the tag
        unsigned char tag[AES_GCM_TAG_SIZE];
        fseek(file, -AES_GCM_TAG_SIZE, SEEK_END);  // Move to the position of the tag
        fread(tag, 1, AES_GCM_TAG_SIZE, file);
        fseek(file, AES_GCM_IV_SIZE, SEEK_SET);  // Move back to the start of the encrypted data

        // Decrypt the file
        unsigned char buffer[1024], outbuf[1024 + EVP_MAX_BLOCK_LENGTH];
        int outlen;
        long remaining = filesize - AES_GCM_IV_SIZE - AES_GCM_TAG_SIZE;  // Subtract the size of the IV and tag
        while (remaining > 0) {
            size_t count = fread(buffer, 1, sizeof(buffer) < remaining ? sizeof(buffer) : remaining, file);
            if (count == 0) break;
            if (EVP_DecryptUpdate(ctx, outbuf, &outlen, buffer, count) != 1) {
                printf("Decryption failed\n");
                return 1;
            }
            fwrite(outbuf, 1, outlen, outfile);
            remaining -= count;
        }

        // Finalize decryption and check the tag
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, AES_GCM_TAG_SIZE, tag) != 1 || EVP_DecryptFinal_ex(ctx, outbuf, &outlen) != 1) {
            printf("Unable to finalize decryption or check tag\n");
            return 1;
        }

        // Clean up
        EVP_CIPHER_CTX_free(ctx);
        fclose(file);
    }

    fclose(outfile);

    return 0;
}
