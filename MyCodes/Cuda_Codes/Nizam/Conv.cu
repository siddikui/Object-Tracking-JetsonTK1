#include<stdio.h>

#define W 16
#define H 16
#define Mask 3

__global__ void Conv(int *input, int *output, int *mask)
{

	int x= blockIdx.x * Mask + threadIdx.x; // Thread Column Index.
	int y= blockIdx.y * Mask + threadIdx.y; // Thread Row Index.
	
	
	int Sum=0;	

	
	for(int i=-1;i<=1;i++)
		for(int j=-1;j<=1;j++)
			Sum+= input[(y+j)*W+(x+i)]*mask[(j+1)*Mask+(i+1)];

	output[y*W+x] = Sum; // Writes result to Output Image pixel.

}

int main(void)
{

	int *h_input, *h_output,*h_mask;
	int *d_input, *d_output, *d_mask;

	

	int SIZE= W*H*sizeof(int);
	
	h_input=(int*)malloc(SIZE);
	h_output=(int*)malloc(SIZE);
	h_mask=(int*)malloc(Mask*Mask*sizeof(int));
	
	for(int k=0; k<Mask*Mask;k++)
	h_mask[k]=1;
	
	for(int i=0;i<W;i++){
		for(int j=0;j<H;j++)
		{
				
		h_input[i*W+j]=1;
		h_output[i*W+j]=0;
				
		}
	}
		
	cudaMalloc(&d_input, SIZE);
	cudaMalloc(&d_output, SIZE);
	cudaMalloc(&d_mask, Mask*Mask*sizeof(int));

	cudaMemcpy(d_input, h_input, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, h_mask, Mask*Mask*sizeof(int), cudaMemcpyHostToDevice);

	dim3 threads(W/Mask, H/Mask);
	dim3 blocks(Mask, Mask);	
	
	Conv<<<blocks, threads>>>(d_input, d_output,d_mask);

	cudaMemcpy(h_output, d_output,SIZE, cudaMemcpyDeviceToHost);

	int y=0;
	for(int i=0; i<W*H; i++)
	{
		if(y==W)
		{
			
		printf("\n");
		
				
		}
		printf("%d ",h_output[i]);
		y++;
	}


	free(h_input);
	free(h_output);
	free(h_mask);

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_mask);

return 0;
}
