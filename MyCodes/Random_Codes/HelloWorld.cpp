// Simple Hello World application. For Jetson TK1.
#include <iostream> 

using namespace std;
int main()
{
	int N=512;    
	for(int i=0;i<N;i++)	
		cout << "Hello Jetson!" << endl;
    return 0;
}
