#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cuda_runtime.h>

using std::cout;
using std::endl;



__global__ void bgr_to_gray_kernel( unsigned char* input, unsigned char* output, int width,
int height,int colorWidthStep,	int grayWidthStep)
{
	
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	
	if((xIndex<width) && (yIndex<height))
	{
		
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
		
		
		const int gray_tid  = yIndex * grayWidthStep + xIndex;

		const unsigned char blue	= input[color_tid];
		const unsigned char green	= input[color_tid + 1];
		const unsigned char red		= input[color_tid + 2];

		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

		output[gray_tid] = static_cast<unsigned char>(gray);
		__syncthreads();
	}
}

void convert_to_gray(const cv::Mat& input, cv::Mat& output)
{
	
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	
	cudaMalloc<unsigned char>(&d_input,colorBytes);
	cudaMalloc<unsigned char>(&d_output,grayBytes);

	
	cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice);

	
	const dim3 block(16,16);

	
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	
	bgr_to_gray_kernel<<<grid,block>>>(d_input,d_output,input.cols,input.rows,input.step,output.step);

	
	cudaDeviceSynchronize();

	
	cudaMemcpy(output.ptr(),d_output,grayBytes,cudaMemcpyDeviceToHost);

	
	cudaFree(d_input);
	cudaFree(d_output);
}



using namespace std;

int main(int argc, char** argv)
{
	
	cv::VideoCapture cap;
	cap.open(0);
	cv:: Mat input;
	

	while(1)
	{
		cap>>input;
		if(!input.data) break;
 		cv::Mat output(input.rows,input.cols,CV_8UC1);
                
		convert_to_gray(input,output);	
		cv::imshow("Input",input);
		cv::imshow("Output",output);		
		if(cv::waitKey(33)>=0) break;
	
	}	

return 0;
}
