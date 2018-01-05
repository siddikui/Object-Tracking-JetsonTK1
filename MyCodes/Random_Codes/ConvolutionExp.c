#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define WIDTH 1024
#define HEIGHT 1024
#define TILE 3

int main(void)
{
	int *Input, *Output_A, *Output_B, *Output_C, *Output_D;
	int *Mask_A, *Mask_B, *Mask_C, *Mask_D;
	int SIZE= WIDTH*HEIGHT*sizeof(int);
	int Row,Col,i,j;

	Input= (int *)malloc(SIZE);

	Output_A= (int *)malloc(SIZE);
	Output_B= (int *)malloc(SIZE);
	Output_C= (int *)malloc(SIZE);
	Output_D= (int *)malloc(SIZE);

	Mask_A= (int *)malloc(TILE*TILE*sizeof(int));
	Mask_B= (int *)malloc(TILE*TILE*sizeof(int));
	Mask_C= (int *)malloc(TILE*TILE*sizeof(int));
	Mask_D= (int *)malloc(TILE*TILE*sizeof(int));

	clock_t Time_Start, Time_End, Time_Difference; // Clock used to measure time for CPU Convolution execution time.
	double Time;

	for(Row=0;Row<WIDTH;Row++)
	for(Col=0;Col<HEIGHT;Col++)
		{
			Input[Row*WIDTH+Col]=1;
			Output_A[Row*WIDTH+Col]=0;	
			Output_B[Row*WIDTH+Col]=0;
			Output_C[Row*WIDTH+Col]=0;
			Output_D[Row*WIDTH+Col]=0;	
		}

	for(Row=0;Row<TILE;Row++)
	for(Col=0;Col<TILE;Col++)
		{
			Mask_A[Row*TILE+Col]=1;
			Mask_B[Row*TILE+Col]=2;
			Mask_C[Row*TILE+Col]=3;
			Mask_D[Row*TILE+Col]=4;
		}

	Time_Start=clock(); // Start Time for CPU Convolution Kernel
	printf ("CPU Executing Convolution Kernel...\n") ;
	printf("\n");

	for(Row=1;Row<WIDTH-1;Row++)
	for(Col=1;Col<HEIGHT-1;Col++)
		{
			int Sum_A=0;
			int Sum_B=0;
			int Sum_C=0;
			int Sum_D=0;

			for(i=-1;i<2;i++)
			for(j=-1;j<2;j++)
				{			
					Sum_A+= Input[(Row+i)*WIDTH+(Col+j)]*Mask_A[(i+1)*TILE+(j+1)];
					Sum_B+= Input[(Row+i)*WIDTH+(Col+j)]*Mask_B[(i+1)*TILE+(j+1)];
					Sum_C+= Input[(Row+i)*WIDTH+(Col+j)]*Mask_C[(i+1)*TILE+(j+1)];
					Sum_D+= Input[(Row+i)*WIDTH+(Col+j)]*Mask_D[(i+1)*TILE+(j+1)];	
				}
		Output_A[Row*WIDTH+Col]=Sum_A;		
		Output_B[Row*WIDTH+Col]=Sum_B;		
		Output_C[Row*WIDTH+Col]=Sum_C;		
		Output_D[Row*WIDTH+Col]=Sum_D;		
		

		}
	
	Time_End=clock();
	Time_Difference=Time_End-Time_Start;
	Time=Time_Difference/(double)CLOCKS_PER_SEC ;	
	
	
	printf ("CPU time for Convolution = %f ms\n", Time*1000) ;
	printf("\n");

	
free(Input);
free(Output_A);
free(Output_B);
free(Output_C);
free(Output_D);
free(Mask_A);
free(Mask_B);
free(Mask_C);
free(Mask_D);

return 0;

}



