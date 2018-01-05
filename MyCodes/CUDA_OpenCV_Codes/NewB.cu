#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cuda_runtime.h>

using std::cout;
using std::endl;

__global__ void bgr_to_gray_kernel( unsigned char* frame, unsigned char* input, unsigned char* output, int width,
int height,int grayWidthStep)
{
	
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;


	if((xIndex>0)  && (yIndex>0) && (xIndex<width) && (yIndex<height))
	{
		
		const int gray_tid  = yIndex * grayWidthStep + xIndex;
		output[gray_tid]=frame[gray_tid]-input[gray_tid];
	}
}

void convert_to_gray(const cv::Mat& frame, const cv::Mat& input, cv::Mat& output)
{
	
	
	const int grayBytes = input.step * input.rows;

	unsigned char *d_frame, *d_input, *d_output;

	
	cudaMalloc<unsigned char>(&d_input,grayBytes);
	cudaMalloc<unsigned char>(&d_frame,grayBytes);
	cudaMalloc<unsigned char>(&d_output,grayBytes);

	
	cudaMemcpy(d_input,input.ptr(),grayBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_frame,frame.ptr(),grayBytes,cudaMemcpyHostToDevice);

	
	const dim3 block(16,16);

	
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	
	bgr_to_gray_kernel<<<grid,block>>>(d_frame,d_input,d_output,input.cols,input.rows,input.step);

	
	cudaDeviceSynchronize();

	
	cudaMemcpy(output.ptr(),d_output,grayBytes,cudaMemcpyDeviceToHost);

	
	cudaFree(d_input);
	cudaFree(d_frame);
	cudaFree(d_output);
}



using namespace std;

int main(int argc, char** argv)
{
	cv:: Mat Input;
	cv:: Mat InputA;
	cv:: Mat InputB;
	cv:: Mat InputC;
	cv:: Mat InputD;	
	cv:: Mat Gray;
	
	//cv:: Mat Frame;	
	
	cv::VideoCapture cap;
	cap.open(0);
	cap>>InputA;
	cap>>InputB;
	cap>>InputC;
	cap>>InputD;

	Input=(InputA+InputB+InputC+InputD)/4;	

	cv::Mat Frame(InputA.rows,InputA.cols,CV_8U);
	
	cvtColor(Input,Frame,CV_BGR2GRAY);	
	


	while(1)
	{
		cap>>Input;
		if(!Input.data) break;
		cv::Mat Gray(Input.rows,Input.cols,CV_8U);
		cvtColor(Input,Gray,CV_BGR2GRAY);

		
		
		cv::Mat Output(Gray.rows,Gray.cols,CV_8U);
		

		convert_to_gray(Frame,Gray,Output);

		cv::Mat binaryA(Gray.rows,Gray.cols,Gray.type());
		cv::threshold(Output, binaryA, 150, 255, cv::THRESH_BINARY);	

		cv::imshow("Input",Gray);
		cv::imshow("Output",binaryA);		
		if(cv::waitKey(33)>=0) break;
	
	}	

return 0;
}
