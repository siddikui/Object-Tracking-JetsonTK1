#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define WIDTH 32
#define HEIGHT 32
#define TILE 3

int main(void)
{
	int *Input, *Output1,*Output2, *Output3, *Output4;
	int *Mask1, *Mask2, *Mask3, *Mask4;
	int SIZE= WIDTH*HEIGHT*sizeof(int);
	int Row,Col,i,j;

	Input= (int *)malloc(SIZE);
	Output1= (int *)malloc(SIZE);
	Output2= (int *)malloc(SIZE);
	Output3= (int *)malloc(SIZE);
	Output4= (int *)malloc(SIZE);
	Mask1= (int *)malloc(TILE*TILE*sizeof(int));
	Mask2= (int *)malloc(TILE*TILE*sizeof(int));
	Mask3= (int *)malloc(TILE*TILE*sizeof(int));
	Mask4= (int *)malloc(TILE*TILE*sizeof(int));


	for(Row=0;Row<WIDTH;Row++)
	for(Col=0;Col<HEIGHT;Col++)
		{
			Input[Row*WIDTH+Col]=1;
			Output1[Row*WIDTH+Col]=0;
			Output2[Row*WIDTH+Col]=0;
			Output3[Row*WIDTH+Col]=0;
			Output4[Row*WIDTH+Col]=0;		
		}
	
	for(Row=0;Row<TILE;Row++) {
	for(Col=0;Col<TILE;Col++) 
		Mask1[Row*TILE+Col]=1;
		Mask2[Row*TILE+Col]=2;
		Mask3[Row*TILE+Col]=3;
		Mask4[Row*TILE+Col]=4;
				}


	for(Row=1;Row<WIDTH-1;Row++)
	for(Col=1;Col<HEIGHT-1;Col++)
		{
			int Sum1=0;
			int Sum2=0;
			int Sum3=0;
			int Sum4=0;
			
			for(i=-1;i<2;i++){
			for(j=-1;j<2;j++)
			Sum1+= Input[(Row+i)*WIDTH+(Col+j)]*Mask1[(i+1)*TILE+(j+1)];
			Sum2+= Input[(Row+i)*WIDTH+(Col+j)]*Mask2[(i+1)*TILE+(j+1)];
			Sum3+= Input[(Row+i)*WIDTH+(Col+j)]*Mask3[(i+1)*TILE+(j+1)];
			Sum4+= Input[(Row+i)*WIDTH+(Col+j)]*Mask4[(i+1)*TILE+(j+1)];
				}
		Output1[Row*WIDTH+Col]=Sum1;		
		Output2[Row*WIDTH+Col]=Sum2; 
		Output3[Row*WIDTH+Col]=Sum3;		
		Output4[Row*WIDTH+Col]=Sum4; 

		}

	for(Row=0;Row<WIDTH;Row++)
	{
		for(Col=0;Col<HEIGHT;Col++)
		printf("%d ",Output1[Row+WIDTH+Col]);
		printf("\n");
		
		
	}
	

	printf("\n");
	printf("\n");	
		

	for(Row=0;Row<WIDTH;Row++)
	{
		for(Col=0;Col<HEIGHT;Col++)
		printf("%d ",Output2[Row+WIDTH+Col]);
		printf("\n");
		
		
	}

	printf("\n");
	printf("\n");

	for(Row=0;Row<WIDTH;Row++)
	{
		for(Col=0;Col<HEIGHT;Col++)
		printf("%d ",Output3[Row+WIDTH+Col]);
		printf("\n");
		
		
	}
	

	printf("\n");
	printf("\n");	
		

	for(Row=0;Row<WIDTH;Row++)
	{
		for(Col=0;Col<HEIGHT;Col++)
		printf("%d ",Output4[Row+WIDTH+Col]);
		printf("\n");
		
		
	}

return 0;

}



