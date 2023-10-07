#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main() {
    clock_t start, end, computation_start, computation_end;
    start = clock();

  
    int x, y;
    const int Width = 800;
    const int Height = 800;
   
    double real, imag;
    const double realMin = -2.5;
    const double realMax = 1.5;
    const double imagMin = -2.0;
    const double imagMax = 2.0;

    /* Calculate pixel dimensions */
    double pixelWidth = (realMax - realMin) / Width;
    double pixelHeight = (imagMax - imagMin) / Height;

    /* Define color components and image file */
    const int maxColorValue = 255;
    FILE *imageFile;
    char *filename = "Mandelbrot.ppm";
    char *comment = "# "; // Comment line in the PPM file

    // RGB color array
    static unsigned char color[3];

    /* Initialize escape radius and maximum iterations */
    const double escapeRadius = 400;
    double escapeRadiusSquared = escapeRadius * escapeRadius;
    const int maxIterations = 1000;

    /* Allocate memory for image data */
    unsigned char *imageData = calloc(3 * Width * Height, sizeof(unsigned char));

    /* Create and open the PPM image file */
    imageFile = fopen(filename, "wb"); /* Open in binary write mode */

    /* Write the PPM file header */
    fprintf(imageFile, "P6\n%s\n%d\n%d\n%d\n", comment, Width, Height, maxColorValue);

    printf("File: %s opened successfully for writing\n", filename);
    printf("Computing the Mandelbrot Set\n");

    /* Get the start time for computation */
    computation_start = clock();

    /* Compute and write image data bytes to the file */
    for (y = 0; y < Height; y++) {
        imag = imagMin + (y * pixelHeight);

        if (fabs(imag) < (pixelHeight / 2)) {
            imag = 0.0; // Main antenna
        }

        for (x = 0; x < Width; x++) {
            real = realMin + (x * pixelWidth);

            /* Initialize orbit and iteration count */
            double zReal = 0.0;
            double zImag = 0.0;
            double zRealSquared = zReal * zReal;
            double zImagSquared = zImag * zImag;

            int iteration = 0;

            /* Iterate to determine if the point is in the set */
            while (iteration < maxIterations && (zRealSquared + zImagSquared) < escapeRadiusSquared) {
                zImag = (2 * zReal * zImag) + imag;
                zReal = zRealSquared - zImagSquared + real;

                zRealSquared = zReal * zReal;
                zImagSquared = zImag * zImag;

                iteration++;
            }

            /* Calculate and assign pixel color based on the iteration count */
            if (iteration == maxIterations) {
                // Point is in the set, mark it as black
                color[0] = 0;
                color[1] = 0;
                color[2] = 0;
            } else {
                // Point is outside the set, assign a color gradient
                double colorFactor = 3 * log((double)iteration) / log((double)(maxIterations - 1));

                if (colorFactor < 1) {
                    color[0] = 0;
                    color[1] = 0;
                    color[2] = (unsigned char)(255 * colorFactor);
                } else if (colorFactor < 2) {
                    color[0] = 0;
                    color[1] = (unsigned char)(255 * (colorFactor - 1));
                    color[2] = 255;
                } else {
                    color[0] = (unsigned char)(255 * (colorFactor - 2));
                    color[1] = 255;
                    color[2] = 255;
                }
            }

            /* Store pixel color in image data */
            imageData[(y * Width * 3) + (x * 3)] = color[0];
            imageData[(y * Width * 3) + (x * 3) + 1] = color[1];
            imageData[(y * Width * 3) + (x * 3) + 2] = color[2];
        }
    }

    /* Get the end time for computation */
    computation_end = clock();

    /* Write image data to the file */
    fwrite(imageData, 1, 3 * Width * Height, imageFile);

   
    double computationTime = ((double)(computation_end - computation_start)) / CLOCKS_PER_SEC;
    printf("Mandelbrot computation process time: %lf seconds\n", computationTime);

   
    fclose(imageFile);

    printf("Completed computing the Mandelbrot Set.\n");
    printf("File: %s successfully closed.\n", filename);

    /* Free allocated memory */
    free(imageData);

    return 0;
}
