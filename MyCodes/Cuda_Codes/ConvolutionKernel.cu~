#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include <time.h>


#define Mask_size 3  //filter size
#define Width 1024   // image width
#define Height 1024   // image height

#define N (Width*Height)

//---------------kernel-------------------

__global__ void ConvolutionKernel (int *I_input, int *Mask1,int *Mask2,int *Mask3,int *Mask4, int *I_output1,int *I_output2,int 												*I_output3,int *I_output4)
{
     	/* Thread Row Index */
	int Row = blockIdx.y * blockDim.y + threadIdx.y;    
    	/* Thread column Index */
	int Col = blockIdx.x * blockDim.x + threadIdx.x; 

    	float value1 = 0;
	float value2 = 0;
	float value3 = 0;
	float value4 = 0;

	int Index = Row*Width+Col; //output Image index
    
/* convolution */

    for(int i=0; i<Mask_size; i++) 
	{
        for(int j=0; j<Mask_size; j++) 
		{
                   int  R_start = i + Row - 1;
                   int  C_start = j + Col - 1;
                         if((C_start>= 0 && C_start < Width) && (R_start>= 0 && R_start < Height))
						 {
                              	value1 += Mask1[i * Mask_size + j] * I_input[R_start* Width + C_start];
				value2 += Mask2[i * Mask_size + j] * I_input[R_start* Width + C_start];
				value1 += Mask2[i * Mask_size + j] * I_input[R_start* Width + C_start];
				value2 += Mask3[i * Mask_size + j] * I_input[R_start* Width + C_start];

						 }
        }
    }
      if((Row < Height) && (Col < Width)) {
            	I_output1[Index] = value1; // convolved image
		I_output2[Index] = value2;
		I_output3[Index] = value3; 
		I_output4[Index] = value4;
    }
}

//----------------------------main-----------------------------------

int main(void)
{
	
//-------------------------------------------------------------------

	int *Image, *Output1,*Output2, *Output3, *Output4;
	int *mask1, *mask2, *mask3, *mask4;
	int SIZE= Width*Height*sizeof(int);
	int Row,Col;

	Image=	(int *)malloc(SIZE);
	Output1= (int *)malloc(SIZE);
	Output2= (int *)malloc(SIZE);
	Output3= (int *)malloc(SIZE);
	Output4= (int *)malloc(SIZE);
	mask1= (int *)malloc(Mask_size*Mask_size*sizeof(int));
	mask2= (int *)malloc(Mask_size*Mask_size*sizeof(int));
	mask3= (int *)malloc(Mask_size*Mask_size*sizeof(int));
	mask4= (int *)malloc(Mask_size*Mask_size*sizeof(int));

//-------------------------------------------------------------------

	int *d_image, *d_mask1,*d_mask2,*d_mask3,*d_mask4,*d_output1, *d_output2,*d_output3, *d_output4; /* pointer to device memory 		
	
												for input image, mask and output */


	//--------------------------------------------------------

	clock_t Time_Start, Time_End, Time_Difference; // Clock used to measure time for CPU Convolution execution time.
	double Time;

	//-----------------------------------------------------------
	
	for(Row=0;Row<Width;Row++)
		for(Col=0;Col<Height;Col++)
		{
			Image[Row*Width+Col]=1;
			Output1[Row*Width+Col]=0;
			Output2[Row*Width+Col]=0;
			Output3[Row*Width+Col]=0;
			Output4[Row*Width+Col]=0;		
		}
	
	//-----------------------------------------------------------

	for(Row=0;Row<Mask_size;Row++) 
		for(Col=0;Col<Mask_size;Col++)
	{ 
		mask1[Row*Mask_size+Col]=1;
		mask2[Row*Mask_size+Col]=2;
		mask3[Row*Mask_size+Col]=3;
		mask4[Row*Mask_size+Col]=4;
	}
		
	//-----------------------------------------------------------
	
	Time_Start=clock(); // Start Time for CPU Convolution Kernel
	printf ("CPU Executing Convolution Kernel...\n") ;
	printf("\n");
	//------------------HOST Execution------------------------
	int i,l;

	for (Row=1;Row<Width-1;Row++)
	for (Col=1;Col<Height-1;Col++)
		{
			int Sum1=0;
			int Sum2=0;
			int Sum3=0;
			int Sum4=0;				
			for(i=-1;i<2;i++)			
				for(l=-1;l<2;l++)
				{

					Sum1 += Image[(Row+i)*Width+(Col+l)]*mask1[(i+1)*Mask_size+(l+1)];
					Sum2 += Image[(Row+i)*Width+(Col+l)]*mask2[(i+1)*Mask_size+(l+1)];
					Sum3 += Image[(Row+i)*Width+(Col+l)]*mask3[(i+1)*Mask_size+(l+1)];
					Sum4 += Image[(Row+i)*Width+(Col+l)]*mask4[(i+1)*Mask_size+(l+1)];
				}
		}


	//-------------------------------------------------------

	Time_End=clock();
	Time_Difference=Time_End-Time_Start;
	Time=Time_Difference/(double)CLOCKS_PER_SEC ;	
	
	printf ("CPU time for Convolution = %f ms\n", Time*1000) ;
	printf("\n");

	//------------------------------------------------------

	/* Device Memory Allocation */
	cudaMalloc(&d_image, (Width*Height)* sizeof(int));
	cudaMalloc(&d_output1, (Width*Height)* sizeof(int));
	cudaMalloc(&d_output2, (Width*Height)* sizeof(int));
	cudaMalloc(&d_output3, (Width*Height)* sizeof(int));
	cudaMalloc(&d_output4, (Width*Height)* sizeof(int));
	cudaMalloc(&d_mask1, (Mask_size*Mask_size)* sizeof(int));
	cudaMalloc(&d_mask2, (Mask_size*Mask_size)* sizeof(int));
	cudaMalloc(&d_mask3, (Mask_size*Mask_size)* sizeof(int));
	cudaMalloc(&d_mask4, (Mask_size*Mask_size)* sizeof(int));
	

	//---------------------------------------------------------

	cudaEvent_t start, stop;       // Cuda API to measure time for Cuda Kernel Execution.
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	
	//--------------------------------------------------------
	
	/*Copying Input Image to GPU Memory */
	cudaMemcpy(d_image, Image, (Width*Height)* sizeof(int), cudaMemcpyHostToDevice); 

	/*Copying Mask to GPU Memory */
	cudaMemcpy(d_mask1, mask1, (Mask_size*Mask_size)* sizeof(int), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_mask2, mask2, (Mask_size*Mask_size)* sizeof(int), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_mask3, mask3, (Mask_size*Mask_size)* sizeof(int), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_mask4, mask4, (Mask_size*Mask_size)* sizeof(int), cudaMemcpyHostToDevice); 
	
	/* Two Dimesional blocks with two dimensional threads */
	dim3 grid(((Width-1)/Mask_size+1),((Height-1)/Mask_size+1));

	/*Number of threads per block is 3x3=9 */	
	dim3 block(Mask_size,Mask_size); 

	//---------------------------------------------

	printf ("GPU Executing Convolution Kernel...\n") ;
	printf("\n");
		
	//--------------------------------------------

	/*Kernel Launch configuration*/	
	ConvolutionKernel <<<block,grid >>>(d_image, d_mask1,d_mask2, d_mask3,d_mask4,d_output1, d_output2,d_output3, d_output4); 
	
	/*copying output Image to Host Memory*/
	cudaMemcpy(Output1, d_output1, (Width*Height)* sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Output2, d_output2, (Width*Height)* sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Output3, d_output3, (Width*Height)* sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Output4, d_output4, (Width*Height)* sizeof(int), cudaMemcpyDeviceToHost);

	//-------------------------------------------
	
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);			// Blocks CPU execution until Device Kernel finishes its job.
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);   	
	
	printf("GPU Execution Time for Convolution Kernel: %fn\n", milliseconds); //GPU Execution Time.
	printf("Effective Bandwidth (GB/s): %fn\n", N*4*2/milliseconds/1e6); 
	//N*4 is the total number of Bytes transferred and (1+1)=2 is for read Input Image and write Output Image.
	printf("\n");

	//------------------------------------------
	
	/* display the convolved image 
	printf("\n");
	printf("the convolved image is\n");
	printf("\n");
	
	int j=0;
	
	for (int i=0;i<(Width*Height);i++)
	{	if(j==Width)
		{	
			printf("\n");
			j=0;
		}
			printf("%d ",Output1[i]);
			j++;
		}

		printf("\n");
		printf("\n");
		printf("\n");

	int x;

	for (int i=0;i<(Width*Height);i++)
	{	if(x==Width)
		{	
			printf("\n");
			x=0;
		}
			printf("%d ",Output2[i]);
			x++;
		}
	
		*/
	
	
	
	cudaFree(d_image);
	cudaFree(d_mask1);
	cudaFree(d_mask2);
	cudaFree(d_mask3);
	cudaFree(d_mask4);
	cudaFree(d_output1);
	cudaFree(d_output2);
	cudaFree(d_output3);
	cudaFree(d_output4);
	
return 0;
}

