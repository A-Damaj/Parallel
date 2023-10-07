#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MASTER 0
#define MAX_WIDTH 32000
#define MAX_HEIGHT 32000
void writePPMImage(const char *filename, unsigned char **pixels, int width, int height) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing.\n");
        return;
    }

    fprintf(file, "P6\n%d %d\n255\n", width, height);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Set pixel color based on the value in pixels[i][j]
            unsigned char color[3];
            double t = (double)pixels[i][j] / 255.0;

            if (t < 1.0 / 3.0) {
                color[0] = 0;
                color[1] = 0;
                color[2] = (unsigned char)(255 * 3 * t);
            } else if (t < 2.0 / 3.0) {
                color[0] = 0;
                color[1] = (unsigned char)(255 * (3 * t - 1));
                color[2] = 255;
            } else {
                color[0] = (unsigned char)(255 * (3 * t - 2));
                color[1] = 255;
                color[2] = 255;
            }

            fwrite(color, sizeof(unsigned char), 3, file);
        }
    }

    fclose(file);
}



struct Complex {
    double real;
    double imag;
};

int calculatePixel(struct Complex c);
int validate(int argc, char *argv[], int *width, int *height);
int isInBounds(int width, int height);
void outputResults(unsigned char **pixels, double time, int width, int height);

int main(int argc, char *argv[]) {
    int comm_sz, my_rank, tag;
    int kill, proc, count, rcvdRow;
    int width, height;
    int row, col, index, greyScaleMod;
    double radius, radSq;
    double startTime, endTime, deltaTime;
    struct Complex c;
    int go;
    unsigned char *pixels;

    radius = 2.0;
    radSq = (radius * radius);
    width = height = 0;
    tag = 0;
    row = 0;
    count = 0;
    kill = -1;
    greyScaleMod = 256;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (!validate(argc, argv, &width, &height)) {
        MPI_Finalize();
        return 0;
    }

    pixels = (unsigned char *)calloc(width, sizeof(unsigned char));

    if (my_rank == MASTER) {
        unsigned char **p = (unsigned char **)malloc(height * sizeof(unsigned char *));
        for (index = 0; index < height; index++) {
            p[index] = (unsigned char *)malloc(width * sizeof(unsigned char));
        }

        printf("width x height = %d x %d\n", width, height);
        startTime = MPI_Wtime();

        proc = 1;
        while (proc < comm_sz) {
            MPI_Send(&row, 1, MPI_INT, proc, tag, MPI_COMM_WORLD);
            proc++;
            row++;
            count++;
        }

        MPI_Status status;
        do {
            MPI_Recv(pixels, width, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            proc = status.MPI_SOURCE;
            rcvdRow = status.MPI_TAG;
            count--;

            if (row < height) {
                MPI_Send(&row, 1, MPI_INT, proc, tag, MPI_COMM_WORLD);
                row++;
                count++;
            } else {
                MPI_Send(&kill, 1, MPI_INT, proc, tag, MPI_COMM_WORLD);
            }

            for (col = 0; col < width; col++) {
                c.real = (col - width / radius) * (radSq / width);
                c.imag = (row - height / radius) * (radSq / width);
                pixels[col] = (calculatePixel(c) * 35) % greyScaleMod;
            }

            for (col = 0; col < width; col++) {
                p[rcvdRow][col] = pixels[col];
            }
        } while (count > 0);

        endTime = MPI_Wtime();
        deltaTime = endTime - startTime;
        outputResults(p, deltaTime, width, height);

        for (index = 0; index < height; index++) {
            free(p[index]);
        }
        free(p);
    } else {
        go = 1;
        while (go) {
            MPI_Recv(&row, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (row == -1) {
                go = 0;
            } else {
                for (col = 0; col < width; col++) {
                    c.real = (col - width / radius) * (radSq / width);
                    c.imag = (row - height / radius) * (radSq / width);
                    pixels[col] = (calculatePixel(c) * 35) % greyScaleMod;
                }
                MPI_Send(pixels, width, MPI_UNSIGNED_CHAR, MASTER, row, MPI_COMM_WORLD);
            }
        }
    }

    free(pixels);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

int calculatePixel(struct Complex c) {
    struct Complex z;
    int count, max;
    double temp, lengthsq;
    max = 256;
    z.real = 0;
    z.imag = 0;
    count = 0;

    do {
        temp = z.real * z.real - z.imag * z.imag + c.real;
        z.imag = 2 * z.real * z.imag + c.imag;
        z.real = temp;
        lengthsq = z.real * z.real + z.imag * z.imag;
        count++;
    } while ((lengthsq < 4.0) && (count < max));

    return count;
}

int isInBounds(int width, int height) {
    if ((width < 0) || (width > MAX_WIDTH) || (height < 0) || (height > MAX_HEIGHT)) {
        printf("ERROR: Invalid image dimensions.\n");
        printf("Height and width range: [1,32000]\n");
        return 0;
    }
    return 1;
}

int validate(int argc, char *argv[], int *width, int *height) {
    if (argc == 3) {
        *width = atoi(argv[1]);
        *height = atoi(argv[2]);
        if (isInBounds(*width, *height)) {
            return 1;
        }
    }
    printf("ERROR: Invalid input.\n");
    printf("Arguments should be <width> <height>\n");
    printf("Shutting down...\n");
    return 0;
}

void outputResults(unsigned char **pixels, double time, int width, int height) {
    unsigned char **p;
    p = pixels; // Remove the const qualifier here
    printf("Set calculation took %.6f s.\n", time);
    printf("Writing image to file 'dynamic.pgm'\n");

    // Write image to file here (implement your file writing logic)

    writePPMImage("dynamic.ppm", pixels, width, height);

    printf("SUCCESS: image written to file.\n");
}

