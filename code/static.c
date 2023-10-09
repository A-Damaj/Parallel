#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>


int main(int argc, char* argv[])
 {
	int iX,iY;
	const int Width = 800; 
	const int Height = 800;

	
	double Cx, Cy;
	const double CxMin = -2.5;
	const double CxMax = 1.5;
	const double CyMin = -2.0;
	const double CyMax = 2.0;
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	//char* list = (char*) malloc (Width*Height*sizeof(char))

	double PixelWidth = (CxMax - CxMin)/Width;
	double PixelHeight = (CyMax - CyMin)/Height;

	/* color component ( R or G or B) is coded from 0 to 255 */
	const int MaxColorComponentValue = 255; 
	FILE * fp;
	char *filename = "Mandelbrot.ppm";
	char *comment = "# ";	

	// RGB color 
	static unsigned char color[3];

	double Zx, Zy;
	double Zx2, Zy2; 
	int Iteration;
	const int IterationMax = 2000;
	const double EscapeRadius = 400;
	double ER2 = EscapeRadius * EscapeRadius;
	
	
	double start, end;
	double cpu_time_used;
	unsigned char row[3 * Width];
	int intervalPerProcess = Height / size;
	int intervalPerProcessRemain = Height % size;
	int updatedsize = size - intervalPerProcessRemain;
	char* list = (char*) malloc ((Width*Height*3)*sizeof(char));

	
	fp = fopen(filename, "wb"); 

	if (rank == 0){
	
		fprintf(fp,"P6\n %s\n %d\n %d\n %d\n", comment, Width, Height, MaxColorComponentValue);
		printf("File: %s successfully opened for writing.\n", filename);
		printf("Computing Mandelbrot Set");

		
		start = MPI_Wtime();
	}

	/* compute and write image data bytes to the file */
	for(iY = rank; iY < Height+updatedsize; iY += size)
	{
		Cy = CyMin + (iY * PixelHeight);
		if (fabs(Cy) < (PixelHeight / 2))
		{
			Cy = 0.0; /* Main antenna */
		}

		for(iX = 0; iX < Width; iX++)
		{
			Cx = CxMin + (iX * PixelWidth);
			/* initial value of orbit = critical point Z= 0 */
			Zx = 0.0;
			Zy = 0.0;
			Zx2 = Zx * Zx;
			Zy2 = Zy * Zy;

			for(Iteration = 0; Iteration < IterationMax && ((Zx2 + Zy2) < ER2); Iteration++)
			{
				Zy = (2 * Zx * Zy) + Cy;
				Zx = Zx2 - Zy2 + Cx;
				Zx2 = Zx * Zx;
				Zy2 = Zy * Zy;
			};

			/* compute  pixel color (24 bit = 3 bytes) */
			if (Iteration == IterationMax)
			{
				// Point within the set. Mark it as black
				color[0] = 0;
				color[1] = 0;
				color[2] = 0;
			}
			else 
			{
				// Point outside the set. Mark it as white
				double c = 3*log((double)Iteration)/log((double)(IterationMax) - 1.0);
				if (c < 1)
				{
					color[0] = 0;
					color[1] = 0;
					color[2] = 255*c;
				}
				else if (c < 2)
				{
					color[0] = 0;
					color[1] = 255*(c-1);
					color[2] = 255;
				}
				else
				{
					color[0] = 255*(c-2);
					color[1] = 255;
					color[2] = 255;
				}
			}
			//Colouring generator
			row[iX*3] = color[0];
			row[iX*3+1] = color[1];
			row[iX*3+2] = color[2];
	}
	if (rank == 0){
		fwrite(row, 1, 3*Width, fp);
		end = MPI_Wtime();
		for (int j=1; j<size; j++){
			MPI_Recv(row,3*Width,MPI_CHAR,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			fwrite(row, 1, 3*Width, fp);

	}

	}	
	else{
	/* write color to the file */
	MPI_Send(row,3*Width,MPI_CHAR,0,iY,MPI_COMM_WORLD);				
			
	}

 	}

	if (rank == 0){
		cpu_time_used = (end - start);
		fclose(fp);
		printf("Completed Computing Mandelbrot Set.\n");
		printf("File: %s successfully closed.\n", filename);
		printf("Mandelbrot computational process time: %lf\n", cpu_time_used);
		fflush(stdout);
 	}
	MPI_Finalize();
	return 0;
 }
