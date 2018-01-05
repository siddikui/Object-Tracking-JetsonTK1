#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>

using std::cout;
using std::endl;

__global__ void bgr_to_gray_kernel( unsigned char* input, unsigned char* output, unsigned char* outputA, unsigned char* outputB, int width, int height, int widthstep)
{
	
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;	

	
	if((xIndex>1)  && (yIndex>1) && (xIndex<width-1) && (yIndex<height-1))
	{				

		const int tid = yIndex * widthstep + xIndex;
		const int tidA = (yIndex) * widthstep + (xIndex-1);
		const int tidB = (yIndex) * widthstep + (xIndex+1);
		const int tidC = (yIndex-1) * widthstep + (xIndex);
		const int tidD = (yIndex-1) * widthstep + (xIndex-1);
		/*const int tidE = (yIndex-1) * widthstep + (xIndex+1);
		const int tidF = (yIndex+1) * widthstep + (xIndex);
		const int tidG = (yIndex+1) * widthstep + (xIndex-1);
		const int tidH = (yIndex+1) * widthstep + (xIndex+1);*/

		if(input[tid]>100)
			outputA[tid]=255;
		else
			outputA[tid]=0;	

		__syncthreads();		
	
		if((outputA[tidA]>=100)&&(outputA[tidB]>=100)&&(outputA[tidC]>=100)&&(outputA[tidD]>=100))
		outputB[tid]=255;
		else
		outputB[tid]=0;
		
		__syncthreads();

		if((outputB[tidA]>=100)||(outputB[tidB]>=100)||(outputB[tidC]>=100)||(outputB[tidD]>=100))
		output[tid]=0;
		else
		output[tid]=255;	
	}
}

void convert_to_gray(const cv::Mat& input, cv::Mat& output)
{
	
	const int Bytes = input.step * input.rows;	

	unsigned char *d_input, *d_output, *d_outputA, *d_outputB;
	
	cudaMalloc<unsigned char>(&d_input,Bytes);
	cudaMalloc<unsigned char>(&d_output,Bytes);
	cudaMalloc<unsigned char>(&d_outputA,Bytes);
	cudaMalloc<unsigned char>(&d_outputB,Bytes);
	
	cudaMemcpy(d_input,input.ptr(),Bytes,cudaMemcpyHostToDevice);
	
	const dim3 block(16,16);
	
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);
	
	bgr_to_gray_kernel<<<grid,block>>>(d_input,d_output,d_outputA,d_outputB,input.cols,input.rows,input.step);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(output.ptr(),d_output,Bytes,cudaMemcpyDeviceToHost);
	
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_outputA);
	cudaFree(d_outputB);

}

int main()
{
	std::string imagePath = "image.jpg";

	
	cv::Mat Input = cv::imread(imagePath,CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat Output(Input.rows,Input.cols,CV_8U);

	if(Input.empty())
	{
		std::cout<<"Image Not Found!"<<std::endl;
		std::cin.get();
		return -1;
	}
	
	convert_to_gray(Input,Output);

	cv::imshow("Input",Input);
	cv::imshow("Output",Output);
	
	cv::waitKey();

	return 0;
}
