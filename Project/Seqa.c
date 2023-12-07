#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> // Include the time.h header
#include <openssl/aes.h>

#define N 100000 // Size of the array

void encryptAES(unsigned char *input, unsigned char *key, unsigned char *output) {
    AES_KEY aesKey;
    AES_set_encrypt_key(key, 128, &aesKey);
    AES_encrypt(input, output, &aesKey);
}

int main() {
    double startTime, endTime;

    unsigned char *data = (unsigned char *)malloc(N * sizeof(unsigned char));
    unsigned char *encryptedData = (unsigned char *)malloc(N * sizeof(unsigned char));
    unsigned char key[16] = "0123456789ABCDEF"; // 128-bit key

    // Sequential version
    startTime = clock() / (double)CLOCKS_PER_SEC; // Start measuring time

    for (int i = 0; i < N; i++) {
        encryptAES(&data[i], key, &encryptedData[i]);
    }

    endTime = clock() / (double)CLOCKS_PER_SEC; // Stop measuring time

    printf("Time taken for sequential encryption: %f seconds\n", endTime - startTime);

    free(data);
    free(encryptedData);

    return 0;
}

