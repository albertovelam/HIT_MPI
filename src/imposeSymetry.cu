#include "turH.h"

static __global__ void normalize_kernel(float2* t1,float2* t2,float2* t3,int IGLOBAL,int NXSIZE)
{

	
	int j  = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
		
	int k=j%NZ;
	j=(j-k)/NZ;

	int h=i*NY*NZ+j*NZ+k;
	
	if(i<NXSIZE &&  j<NY && k<NZ )
	{

	
	float N3=(float)(N*N*N);	
	
	t1[h].x/=N3;
	t2[h].x/=N3;
	t3[h].x/=N3;

	t1[h].y/=N3;
	t2[h].y/=N3;
	t3[h].y/=N3;

		


	}

}


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;
static int threadsPerBlock_in=16;
static cudaError_t ret;

// Functino to turn to zero all those modes dealiased

extern void imposeSymetry(vectorField t)
{
	
	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.y=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.x=NY*NZ/threadsPerBlock.y;


	fftBackward(t.x);
	fftBackward(t.y);
	fftBackward(t.z);	

	normalize_kernel<<<blocksPerGrid,threadsPerBlock>>>(t.x,t.y,t.z,IGLOBAL,NXSIZE);
	kernelCheck(ret,"dealias",1);

	fftForward(t.x);
	fftForward(t.y);
	fftForward(t.z);	

	return;

}


