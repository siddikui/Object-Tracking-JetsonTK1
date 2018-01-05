#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;

int main(int argc, char** argv)
{
	cv::namedWindow("Video Display",cv::WINDOW_AUTOSIZE);

	cv::VideoCapture capA;
	cv::VideoCapture capB;

	capA.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capA.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	capB.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capB.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	capA.open(0);
	capB.open(1);

	cv:: Mat frameA;
	cv:: Mat frameB;

	while(1)
	{
		capA>>frameA;
		if(!frameA.data) break;

		capB>>frameB;
		if(!frameB.data) break;
		
		cv::imshow("Video Display A",frameA);
		cv::imshow("Video Display A",frameB);
		if(cv::waitKey(33)>=0) break;
	
	}	

return 0;
}

