#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cuda_runtime.h>
#include<time.h>

using std::cout;
using std::endl;

__global__ void Kernel(unsigned char* FrameA,unsigned char* FrameB,unsigned char* Frame,unsigned char* Gray,unsigned char* Bin,unsigned char* Ero,unsigned char* Dil,unsigned char* ExA,unsigned char* ExB,unsigned char* ExC,unsigned char* ExD,unsigned char* ExE,unsigned char* FrameF,int width,int height,int colorWidthStep, int grayWidthStep)
{
	
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	
	if((xIndex>1)  && (yIndex>1) && (xIndex<width-1) && (yIndex<height-1))
	{
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
		const int gray_tid  = yIndex * grayWidthStep + xIndex;

		Frame[color_tid]=FrameB[color_tid]-FrameA[color_tid];	
		Frame[color_tid+1]=FrameB[color_tid+1]-FrameA[color_tid+1];	
		Frame[color_tid+2]=FrameB[color_tid+2]-FrameA[color_tid+2];	
	
		__syncthreads();

		const unsigned char blue	= Frame[color_tid];
		const unsigned char green	= Frame[color_tid + 1];
		const unsigned char red		= Frame[color_tid + 2];

		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

		Gray[gray_tid] = static_cast<unsigned char>(gray);	

		__syncthreads();

		if(Gray[gray_tid]>220)
			Bin[gray_tid]=255;
		else
			Bin[gray_tid]=0;

		__syncthreads();	

		const int tidA = (yIndex) * grayWidthStep + (xIndex); 		// x , y
		const int tidB = (yIndex-1) * grayWidthStep + (xIndex); 	// x , y-1
		const int tidC = (yIndex+1) * grayWidthStep + (xIndex); 	// x , y+1
		const int tidD = (yIndex) * grayWidthStep + (xIndex-1);		// x-1 , y
		const int tidE = (yIndex) * grayWidthStep + (xIndex+1);		// x+1 , y	
		const int tidF = (yIndex-1) * grayWidthStep + (xIndex-1);	// x-1 , y-1
		const int tidG = (yIndex-1) * grayWidthStep + (xIndex+1);	// x+1 , y-1
		const int tidH = (yIndex+1) * grayWidthStep + (xIndex-1);	// x-1 , y+1	
		const int tidI = (yIndex+1) * grayWidthStep + (xIndex+1);	// x+1 , y+1		
		
		const int tidJ = (yIndex) * grayWidthStep + (xIndex-2);		// x-2 , y
		const int tidK = (yIndex) * grayWidthStep + (xIndex+2);		// x+2 , y	
		const int tidL = (yIndex-1) * grayWidthStep + (xIndex-2);	// x-2 , y-1
		const int tidM = (yIndex-1) * grayWidthStep + (xIndex+2);	// x+2 , y-1
		const int tidN = (yIndex+1) * grayWidthStep + (xIndex-2);	// x-1 , y+1	
		const int tidO = (yIndex+1) * grayWidthStep + (xIndex+2);	// x+1 , y+1		
				
		const int tidP = (yIndex-2) * grayWidthStep + (xIndex-1);	// x-1 , y-1
		const int tidQ = (yIndex-2) * grayWidthStep + (xIndex);		// x   , y-2	
		const int tidR = (yIndex-2) * grayWidthStep + (xIndex+1);	// x+1 , y-2
		const int tidS = (yIndex+2) * grayWidthStep + (xIndex-1);	// x+2 , y-1
		const int tidT = (yIndex+2) * grayWidthStep + (xIndex);		// x-1 , y+1	
		const int tidU = (yIndex+2) * grayWidthStep + (xIndex+1);	// x+1 , y+1		
						



/**/		
	
		if((Bin[tidA]>100)&&(Bin[tidB]>100)&&(Bin[tidD]>100)&&(Bin[tidE]>100)&&(Bin[tidG]>100)&&(Bin[tidF]>100))
		Ero[gray_tid]=255;
		else
		Ero[gray_tid]=0;
		
		__syncthreads();

		if((Ero[tidA]>100)&&(Ero[tidB]>100)&&(Ero[tidD]>100)&&(Ero[tidE]>100)&&(Ero[tidG]>100)&&(Ero[tidF]>100))
		Dil[gray_tid]=255;
		else
		Dil[gray_tid]=0;
		
		__syncthreads();

		if((Dil[tidA]>100)&&(Dil[tidB]>100)&&(Dil[tidD]>100)&&(Dil[tidE]>100)&&(Dil[tidG]>100)&&(Dil[tidF]>100))
		ExA[gray_tid]=255;
		else
		ExA[gray_tid]=0;

		__syncthreads();

		if((ExA[tidA]>100)||(ExA[tidB]>100)||(ExA[tidC]>100)||(ExA[tidD]>100)||(ExA[tidE]>100)||
		(ExA[tidF]>100)||(ExA[tidG]>100)||(ExA[tidH]>100)||(ExA[tidI]>100)||(ExA[tidJ]>100)||(ExA[tidK]>100)
		||(ExA[tidL]>100)||(ExA[tidM]>100)||(ExA[tidN]>100)||(ExA[tidO]>100))
		ExB[gray_tid]=255;
		else
		ExB[gray_tid]=0;

		__syncthreads();

		if((ExB[tidA]>100)||(ExB[tidB]>100)||(ExB[tidC]>100)||(ExB[tidD]>100)||(ExB[tidE]>100)||
		(ExB[tidF]>100)||(ExB[tidG]>100)||(ExB[tidH]>100)||(ExB[tidI]>100)||(ExB[tidJ]>100)||(ExB[tidK]>100)
		||(ExB[tidL]>100)||(ExB[tidM]>100)||(ExB[tidN]>100)||(ExB[tidO]>100))
		ExC[gray_tid]=255;
		else
		ExC[gray_tid]=0;

		__syncthreads();

		if((ExC[tidA]>100)||(ExC[tidB]>100)||(ExC[tidC]>100)||(ExC[tidD]>100)||(ExC[tidE]>100)||
		(ExC[tidF]>100)||(ExC[tidG]>100)||(ExC[tidH]>100)||(ExC[tidI]>100)||(ExC[tidJ]>100)||(ExC[tidK]>100)
		||(ExC[tidL]>100)||(ExC[tidM]>100)||(ExC[tidN]>100)||(ExC[tidO]>100))
		ExD[gray_tid]=255;
		else
		ExD[gray_tid]=0;

		__syncthreads();

		if((ExD[tidA]>100)||(ExD[tidB]>100)||(ExD[tidC]>100)||(ExD[tidD]>100)||(ExD[tidE]>100)||
		(ExD[tidF]>100)||(ExD[tidG]>100)||(ExD[tidH]>100)||(ExD[tidI]>100)||(ExD[tidP]>100)||(ExD[tidQ]>100)
		||(ExD[tidR]>100)||(ExD[tidS]>100)||(ExD[tidT]>100)||(ExD[tidU]>100))
		ExE[gray_tid]=255;
		else
		ExE[gray_tid]=0;

		__syncthreads();

		if((ExE[tidA]>100)||(ExE[tidB]>100)||(ExE[tidC]>100)||(ExE[tidD]>100)||(ExE[tidE]>100)||
		(ExE[tidF]>100)||(ExE[tidG]>100)||(ExE[tidH]>100)||(ExE[tidI]>100)||(ExE[tidP]>100)||(ExE[tidQ]>100)
		||(ExE[tidR]>100)||(ExE[tidS]>100)||(ExE[tidT]>100)||(ExE[tidU]>100))
		FrameF[gray_tid]=255;
		else
		FrameF[gray_tid]=0;

		

					
		
	}
}


void ImageGrayConverter(const cv::Mat& FrameA, const cv::Mat& FrameB, cv::Mat& FrameFinal)
{
	
	const int colorBytes = FrameA.step * FrameA.rows;
	const int grayBytes = FrameFinal.step * FrameFinal.rows;

	unsigned char *D_FrameA, *D_FrameB, *D_Frame;

	unsigned char *D_Gray, *D_Bin, *D_Ero, *D_Dil, *D_ExA, *D_ExB, *D_ExC,*D_ExD,*D_ExE, *D_FrameFinal;
	
	cudaMalloc<unsigned char>(&D_FrameA,colorBytes);
	cudaMalloc<unsigned char>(&D_FrameB,colorBytes);
	cudaMalloc<unsigned char>(&D_Frame,colorBytes);

	cudaMalloc<unsigned char>(&D_Gray,grayBytes);
	cudaMalloc<unsigned char>(&D_Bin,grayBytes);
	cudaMalloc<unsigned char>(&D_Ero,grayBytes);
	cudaMalloc<unsigned char>(&D_Dil,grayBytes);
	cudaMalloc<unsigned char>(&D_ExA,grayBytes);
	cudaMalloc<unsigned char>(&D_ExB,grayBytes);
	cudaMalloc<unsigned char>(&D_ExC,grayBytes);
	cudaMalloc<unsigned char>(&D_ExD,grayBytes);
	cudaMalloc<unsigned char>(&D_ExE,grayBytes);
		
	cudaMalloc<unsigned char>(&D_FrameFinal,grayBytes);
	
	cudaMemcpy(D_FrameA, FrameA.ptr(),colorBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(D_FrameB, FrameB.ptr(),colorBytes,cudaMemcpyHostToDevice);
		
	const dim3 block(32,32);
	
	const dim3 grid((FrameA.cols + block.x - 1)/block.x, (FrameA.rows + block.y - 1)/block.y);
	
	Kernel<<<grid,block>>>(D_FrameA,D_FrameB,D_Frame,D_Gray,D_Bin,D_Ero,D_Dil,D_ExA,D_ExB,D_ExC,D_ExD,D_ExE,D_FrameFinal,
				FrameA.cols,FrameA.rows,FrameA.step,FrameFinal.step);
	
	cudaDeviceSynchronize();	
	
	cudaMemcpy(FrameFinal.ptr(),D_FrameFinal,grayBytes,cudaMemcpyDeviceToHost);
	
	cudaFree(D_FrameA);
	cudaFree(D_FrameB);
	cudaFree(D_Frame);
	cudaFree(D_Gray);
	cudaFree(D_Bin);
	cudaFree(D_Ero);
	cudaFree(D_Dil);
	cudaFree(D_ExA);
	cudaFree(D_ExB);
	cudaFree(D_ExC);
	cudaFree(D_ExD);
	cudaFree(D_ExE);			
	cudaFree(D_FrameFinal);
}

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat InputA;	
	Mat InputB;

	cv::VideoCapture cap;
	cap.open(0); 

	clock_t Time_Start, Time_End, Time_Difference; // Clock used to measure time for CPU 
	double Time;						

	while(1)
	{	
		Time_Start=clock();

		cap>>InputA;
		cap>>InputB;				
     
		cv::Mat FrameFinal(InputA.rows,InputA.cols,CV_8U);

		ImageGrayConverter(InputA,InputB,FrameFinal);

		Time_End=clock();
		Time_Difference=Time_End-Time_Start;
		Time=Time_Difference/(double)CLOCKS_PER_SEC ;
		printf ("GPU Frame Rate = %f FPS\n",1/Time);
			
		RNG rng(12345);
		vector<vector<Point> >contours;
  		vector<Vec4i>hierarchy;

		findContours(FrameFinal,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0, 0));

		vector<vector<Point> > contours_poly(contours.size());
 		vector<Rect>boundRect(contours.size());
		vector<Point2f>center(contours.size());
 		vector<float>radius(contours.size());

  		for(int i=0;i<contours.size();i++)
			{ 
				approxPolyDP(Mat(contours[i]),contours_poly[i],3,true);
				boundRect[i]=boundingRect(Mat(contours_poly[i]));
				minEnclosingCircle((Mat)contours_poly[i],center[i],radius[i]);
			}


		for(int i=0;i<contours.size();i++)
			{
				Scalar color = Scalar(0,0,255);
				rectangle(InputA,boundRect[i].tl(),boundRect[i].br(),color,2,8,0);
			}
			
		cv::imshow("GPU Accelerated Tracking",InputA);	
		cv::imshow("Prcessed Frame",FrameFinal);		
		
		if(cv::waitKey(33)>=0) break;	
	}	
	return 0;
}
