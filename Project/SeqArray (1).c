#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/err.h>

#define AES_GCM_KEY_SIZE 16
#define AES_GCM_IV_SIZE 12
#define AES_GCM_TAG_SIZE 16

// Function to handle OpenSSL errors
void handleErrors(void) {
    ERR_print_errors_fp(stderr);
    abort();
}

int main() {
    // Generate a sample array of size N filled with random data
    long  N = 9999999999; // Size of the sample array
    unsigned char *plaintext = malloc(N);
    if (RAND_bytes(plaintext, N) != 1) {
        handleErrors();
    }

    // Initialize the encryption context
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        handleErrors();
    }

    // Initialize AES-GCM with 128-bit key
    unsigned char key[AES_GCM_KEY_SIZE] = {0x54, 0x68, 0x69, 0x73, 0x49, 0x73, 0x41, 0x53, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x4b, 0x65, 0x79}; // key in hex
    unsigned char iv[AES_GCM_IV_SIZE];
    if (RAND_bytes(iv, AES_GCM_IV_SIZE) != 1) {
        handleErrors();
    }

    // Set the key and IV for encryption
    if (EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, key, iv) != 1) {
        handleErrors();
    }

    // Encrypt the sample array
    unsigned char ciphertext[N + AES_GCM_TAG_SIZE];
    int len;
    if (EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, N) != 1) {
        handleErrors();
    }

    // Finalize encryption and get the tag
    unsigned char tag[AES_GCM_TAG_SIZE];
    if (EVP_EncryptFinal_ex(ctx, ciphertext + len, &len) != 1) {
        handleErrors();
    }
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, AES_GCM_TAG_SIZE, tag) != 1) {
        handleErrors();
    }

    // Clean up encryption context
    EVP_CIPHER_CTX_free(ctx);

    // Initialize the decryption context
    ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        handleErrors();
    }

    // Set the key and IV for decryption
    if (EVP_DecryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, key, iv) != 1) {
        handleErrors();
    }

    // Set the expected tag value
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, AES_GCM_TAG_SIZE, tag) != 1) {
        handleErrors();
    }

    // Decrypt the ciphertext
    unsigned char decryptedtext[N];
    if (EVP_DecryptUpdate(ctx, decryptedtext, &len, ciphertext, N) != 1) {
        handleErrors();
    }

    // Finalize decryption
    if (EVP_DecryptFinal_ex(ctx, decryptedtext + len, &len) != 1) {
        printf("Decryption failed. Incorrect tag?\n");
        handleErrors();
    }

    // Clean up decryption context
    EVP_CIPHER_CTX_free(ctx);

    // Display the original, encrypted, and decrypted data
    // printf("Original Data:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%02x", plaintext[i]);
    // }
    // printf("\n\nEncrypted Data:\n");
    // for (int i = 0; i < N + AES_GCM_TAG_SIZE; i++) {
    //     printf("%02x", ciphertext[i]);
    // }
    // printf("\n\nDecrypted Data:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%02x", decryptedtext[i]);
    // }
    // printf("\n");

    // Free allocated memory
    free(plaintext);

    return 0;
}
