/* 
Program Name: VectorAdd
This program uses multiple blocks and multiple threads
for the same vector addition.
*/
#include<stdio.h>
#include<time.h>

__global__ void VectorAdd(int *a, int *b, int *c)
{
	int id= threadIdx.x + blockIdx.x*blockDim.x;
	c[id]=a[id]*b[id];	
}
#define N 8192*4096
#define THREADS_PER_BLOCK 1024
int main()
{
	int *a, *b, *c, *d;
	int *d_a, *d_b, *d_c;	

	a= (int *)malloc(N*sizeof(int));
	b= (int *)malloc(N*sizeof(int));
	c= (int *)malloc(N*sizeof(int));
	d= (int *)malloc(N*sizeof(int));

	cudaMalloc(&d_a,N*sizeof(int));
	cudaMalloc(&d_b,N*sizeof(int));
	cudaMalloc(&d_c,N*sizeof(int));

	for(int i=0; i<N; i++)
	{
		a[i]=i;	b[i]=i;		
	}	

	clock_t Time_Start, Time_End, Time_Difference; // Clock used to measure time for CPU Vector Addition execution time.
	double Time;

	Time_Start=clock(); // Start Time for CPU Vector Addition Kernel
	printf ("CPU Executing Vector Add Kernel...\n") ;
	printf("\n");
	
	for(int i=0; i<N; i++)
	{
		d[i]=a[i]+b[i];		
	}

	Time_End=clock();
	Time_Difference=Time_End-Time_Start;
	Time=Time_Difference/(double)CLOCKS_PER_SEC ;

	printf ("CPU time for Addition = %f ms\n", Time*1000) ;
	printf("\n");

	cudaEvent_t start, stop;       // Cuda API to measure time for Cuda Kernel Execution.
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
	
	cudaEventRecord(start);
	
	cudaMemcpy(d_a,a,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,N*sizeof(int),cudaMemcpyHostToDevice);

	
	printf ("GPU Executing Convolution Kernel...\n") ;
	printf("\n");
	
	VectorAdd<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a,d_b,d_c);
	
	
	cudaMemcpy(c,d_c,N*sizeof(int),cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);	//Blocks CPU execution until Device Kernel finishes its job.
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);  

	printf("GPU Execution Time for Convolution Kernel: %fn\n", milliseconds); //GPU Execution Time.
	printf("Effective Bandwidth (GB/s): %fn\n", N*4*2/milliseconds/1e6); 
	//N*4 is the total number of Bytes transferred and (1+1)=2 is for read Input Image and write Output Image.
	printf("\n");

	for(int j=0;j<10;j++)
		printf("%d\n",c[j]);
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(a);
	free(b);
	free(c);
	free(d);
	return 0;
}
