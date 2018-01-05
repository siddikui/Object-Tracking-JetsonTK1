#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define WIDTH 32
#define HEIGHT 32
#define TILE 3

int main(void)
{
	int *Input, *Output;
	int *Mask;
	int SIZE= WIDTH*HEIGHT*sizeof(int);
	int Row,Col,i,j;

	Input= (int *)malloc(SIZE);
	Output= (int *)malloc(SIZE);
	Mask= (int *)malloc(TILE*TILE*sizeof(int));

	for(Row=0;Row<WIDTH;Row++)
	for(Col=0;Col<HEIGHT;Col++)
		{
			Input[Row*WIDTH+Col]=1;
			Output[Row*WIDTH+Col]=0;		
		}
	for(Row=0;Row<TILE;Row++)
	for(Col=0;Col<TILE;Col++)
		Mask[Row*TILE+Col]=1;

	for(Row=1;Row<WIDTH-1;Row++)
	for(Col=1;Col<HEIGHT-1;Col++)
		{
			int Sum=0;
			
			for(i=-1;i<2;i++)
			for(j=-1;j<2;j++)
			Sum+= Input[(Row+i)*WIDTH+(Col+j)]*Mask[(i+1)*TILE+(j+1)];

		Output[Row*WIDTH+Col]=Sum;		
		

		}

	for(Row=0;Row<WIDTH;Row++)
	{
		for(Col=0;Col<HEIGHT;Col++)
		printf("%d ",Output[Row+WIDTH+Col]);
		printf("\n");
		
	}
	
return 0;

}



