#include "turH.h"


static __global__ void shift_kernel(float2* tx,float2* ty,float2* tz,float Delta_1,float Delta_2,float Delta_3,int IGLOBAL,int NXSIZE)
{

	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
		
	int k=j%NZ;
	j=(j-k)/NZ;

	float k1,k2,k3;
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;
	

	int h=i*NY*NZ+j*NZ+k;
	
	if(i<NXSIZE &&  j<NY && k<NZ )
	{

	float2 t1=tx[h];
	float2 t2=ty[h];
	float2 t3=tz[h];
	
	float aux_x;
	float aux_y;

	// Phase shifting by Delta;

	float sine=sin(k1*Delta_1+k2*Delta_2+k3*Delta_3);
	float cosine=cos(k1*Delta_1+k2*Delta_2+k3*Delta_3);
	
	//t1;

	aux_x=cosine*t1.x-sine*t1.y;
	aux_y=sine*t1.x+cosine*t1.y;

	t1.x=aux_x;
	t1.y=aux_y;
	
	//t2;

	aux_x=cosine*t2.x-sine*t2.y;
	aux_y=sine*t2.x+cosine*t2.y;

	t2.x=aux_x;
	t2.y=aux_y;	

	//t3	
	
	aux_x=cosine*t3.x-sine*t3.y;
	aux_y=sine*t3.x+cosine*t3.y;

	t3.x=aux_x;
	t3.y=aux_y;	
	
	
	tx[h]=t1;
	ty[h]=t2;
	tz[h]=t3;



	}

}


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


extern void shift(vectorField t,float* Delta)
{


	//SET BLOCK DIMENSIONS
	
	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NY*NZ/threadsPerBlock.y;


	shift_kernel<<<blocksPerGrid,threadsPerBlock>>>(t.x,t.y,t.z,Delta[0],Delta[1],Delta[2],IGLOBAL,NXSIZE);
	kernelCheck(RET,"dealias",1);
	
	return;

}


