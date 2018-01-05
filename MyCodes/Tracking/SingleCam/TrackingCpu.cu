#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cuda_runtime.h>
#include<time.h>


using namespace std;
using namespace cv;

void FrameProcessor(const cv::Mat& InputA, const cv::Mat& InputB, cv::Mat& FrameFinal)
{	

	int erosion_size=0;
	int d_size=2;
	
	FrameFinal=InputA-InputB;		
		
	cvtColor(FrameFinal,FrameFinal,CV_BGR2GRAY);

	threshold(FrameFinal,FrameFinal,100,255,THRESH_BINARY);

	Mat element = getStructuringElement(MORPH_RECT,Size(2*erosion_size+1,2*erosion_size+1),
                                       			Point(erosion_size,erosion_size));

 	erode(FrameFinal,FrameFinal,element);
	erode(FrameFinal,FrameFinal,element);
	erode(FrameFinal,FrameFinal,element);

	Mat element1 = getStructuringElement(MORPH_RECT,Size(3*d_size+1,3*d_size+1),
                                       			Point(d_size,d_size));

	dilate(FrameFinal,FrameFinal,element1);
	dilate(FrameFinal,FrameFinal,element1);
	dilate(FrameFinal,FrameFinal,element1);
	dilate(FrameFinal,FrameFinal,element1);
	dilate(FrameFinal,FrameFinal,element1);
	dilate(FrameFinal,FrameFinal,element1);
	dilate(FrameFinal,FrameFinal,element1);
	

}

void DrawContour(const cv::Mat& FrameFinal, cv::Mat& Input)
{
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
				rectangle(Input,boundRect[i].tl(),boundRect[i].br(),color,2,8,0);
			}					

}

int main(int argc, char** argv)
{
	       
	cv::VideoCapture capA;
	
	capA.open(1); 	
	 	

	clock_t Time_Start, Time_End, Time_Difference; 
	double Time;

	Mat InputA;	
	Mat InputB;

	Mat FrameFinalA; 
	 
	/*
	cv::VideoCapture cap;
	cap.open(string(argv[1]));*/

	
	for(;;)
    	{      
	
	
	capA>>InputA;
	capA>>InputB;

	Time_Start=clock();

	FrameProcessor(InputA,InputB,FrameFinalA);	

	Time_End=clock();
	Time_Difference=Time_End-Time_Start;
	Time=Time_Difference/(double)CLOCKS_PER_SEC ;
	printf ("CPU Frame Rate = %f FPS\n",1/Time);

	DrawContour(FrameFinalA,InputA);
	
	imshow( "CPU Tracking", InputA);		
      
        if(waitKey(30)>=0) break;
    }
   
    return 0;
}
