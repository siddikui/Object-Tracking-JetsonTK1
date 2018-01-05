/* 

 Parallel Processing Architecture and Algorithms, Spring-2015.
 Project: Image Convolution with Cuda.
 Muhammad Shahid Noman Siddiqui. 
 Sp-2014/M.Sc.CE/007   

 Note: The following heterogeneous code has been developed on Intel core i7, 2.8GHz processor with
 Nvidia NVS3100m notebook business graphic card with 16 Cuda Cores. Visual Studio 2010 and Cuda 
 Toolkit 6.5 has been used. Plus this code is a simple display of the 2D convolution i.e it is a 
 test code for very small image size and it doesn't yet take into account the real image. 
 
*/

/* 
Program Name: Cuda_2DConvolution
This program has the CUDA only code for Convolution.
In future, this code can be mixed up with MPI and run 
on CASE Cluster System. Plus this is only an illustration
of Convolution with a simple impulse filter. 
*/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define Width  1024   //Image Width and Height.
#define Height 1024
#define Tile 32

#define N (Width*Height)

// Device Side Convolution Function and is callable from Host. 

__global__ void Image_Convolution(int *Input, int *Output)
{
	int x= blockIdx.x * Tile + threadIdx.x; // Thread Column Index.
	int y= blockIdx.y * Tile + threadIdx.y; // Thread Row Index.
	
	int Mask[3][3]={{1,1,1},{1,1,1},{1,1,1}}; // Impulse Filter.
	
	int Sum=0;	
	/* Each pixel of Image has been mapped to each thread and 
	multiplied by corresponding neighbouring Filter coefficients.
	*/
	for(int i=-1;i<=1;i++)
		for(int j=-1;j<=1;j++)
			Sum+= Input[(y+j)*Width+(x+i)]*Mask[j+1][i+1];

	Output[y*Width+x] = Sum; // Writes result to Output Image pixel.
}

int main(void)
{
	int *I_Image, *O_Image; // Host variables for Input and Output Images.
	int *dev_I_Image, *dev_O_Image; // Device side pointers to Input and Output Images.
	
	int Mask[3][3]={{1,1,1},{1,1,1},{1,1,1}}; // Impulse Filter.
	
	

	int SIZE=Width*Height*sizeof(int); 

	I_Image=(int *)malloc(SIZE);	
	O_Image=(int *)malloc(SIZE);

	clock_t Time_Start, Time_End, Time_Difference; // Clock used to measure time for CPU Convolution execution time.
	double Time;
	
	for(int i=0;i<Width;i++) // Image has been initialized with value 1 for all pixels.
		for(int j=0;j<Height;j++)
		{
			
			I_Image[i*Width+j]=1;
		}


	Time_Start=clock(); // Start Time for CPU Convolution Kernel
	printf ("CPU Executing Convolution Kernel...\n") ;
	printf("\n");

	for (int row=1;row<Height-1;row++) // CPU Kernel for Convolution.
		 for (int col=1;col<Width-1;col++) // Avoiding Memory access beyond the Image bounds.
		 {
			int Sum= 0;			 
	
			for (int i=-1;i<=1;i++)
				for (int j=-1;j<=1;j++)									
					Sum += I_Image[i*row+j]*Mask[1+i][1+j];				
			 
  
		}
	
	Time_End=clock();
	Time_Difference=Time_End-Time_Start;
	Time=Time_Difference/(double)CLOCKS_PER_SEC ;	
	
	
	printf ("CPU time for Convolution = %f ms\n", Time*1000) ;
	printf("\n");

	cudaMalloc(&dev_I_Image,SIZE); // Allocating memory onto the GPU.
	cudaMalloc(&dev_O_Image,SIZE);	

	cudaEvent_t start, stop;       // Cuda API to measure time for Cuda Kernel Execution.
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	cudaMemcpy(dev_I_Image,I_Image,SIZE,cudaMemcpyHostToDevice); // Copying Input Image to GPU Memory.

	dim3 dimGrid(Width/Tile,Height/Tile); // Two Dimesional blocks with two dimensional threads.
	dim3 dimBlock(Tile,Tile);             // 16*16=256, max number of threads per block is 512. 

	
	printf ("GPU Executing Convolution Kernel...\n") ;
	printf("\n");
	
	Image_Convolution<<<dimGrid,dimBlock>>>(dev_I_Image,dev_O_Image); // Kernel Launch configuration.
	
	cudaMemcpy(O_Image,dev_O_Image,SIZE,cudaMemcpyDeviceToHost); // Copying Output Image back to Host Memory.

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);			// Blocks CPU execution until Device Kernel finishes its job.
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);   	
	
	printf("GPU Execution Time for Convolution Kernel: %fn\n", milliseconds); //GPU Execution Time.
	printf("Effective Bandwidth (GB/s): %fn\n", N*4*2/milliseconds/1e6); 
	//N*4 is the total number of Bytes transferred and (1+1)=2 is for read Input Image and write Output Image.
	printf("\n");

	cudaFree(dev_I_Image); // Since we are good coders, freeing device memory to keep the atmosphere clean. :p
	cudaFree(dev_O_Image);

	free(I_Image);
	free(O_Image);
	
	return 0;
}
