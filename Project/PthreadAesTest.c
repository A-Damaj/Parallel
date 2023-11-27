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
    int  outfile;
    long start;
    long end;
} ThreadData;

void *encrypt_chunk(void *arg) {
    ThreadData *data = (ThreadData *)arg;

    // Your existing encryption code here, modified to read from data->file
    // starting at data->start and ending at data->end, and write to data->outfile





      FILE *outfile = fopen("encrypted_file"+"", "wb");
    if (!outfile) {
        perror("Unable to open output file");
        return 1;
    }
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

    // Open the output file
    FILE *outfile = fopen("encrypted_file", "wb");
    if (!outfile) {
        perror("Unable to open output file");
        return 1;
    }

    // Create the threads
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].file = file;
        thread_data[i].outfile = i;
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
    fclose(outfile);

    return 0;
}
